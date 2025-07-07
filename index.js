import {
  eventSource,
  event_types,
  extension_prompt_roles,
  extension_prompt_types,
  getCurrentChatId,
  getRequestHeaders,
  is_send_press,
  saveSettingsDebounced,
  setExtensionPrompt,
  substituteParams,
  substituteParamsExtended,
} from '../../../../script.js';
import { getDataBankAttachments, getDataBankAttachmentsForSource, getFileAttachment } from '../../../chats.js';
import { debounce_timeout } from '../../../constants.js';
import {
  ModuleWorkerWrapper,
  extension_settings,
  getContext,
  renderExtensionTemplateAsync,
} from '../../../extensions.js';
import { POPUP_RESULT, POPUP_TYPE, callGenericPopup } from '../../../popup.js';
import { registerDebugFunction } from '../../../power-user.js';
import { SlashCommand } from '../../../slash-commands/SlashCommand.js';
import { SlashCommandParser } from '../../../slash-commands/SlashCommandParser.js';
import { textgen_types, textgenerationwebui_settings } from '../../../textgen-settings.js';
import {
  debounce,
  getStringHash,
  onlyUnique,
  splitRecursive,
  trimToEndSentence,
  trimToStartSentence,
  waitUntilCondition,
} from '../../../utils.js';
import { getSortedEntries } from '../../../world-info.js';

/**
 * @typedef {object} HashedMessage
 * @property {string} text - The hashed message text
 * @property {number} hash - The hash used as the vector key
 * @property {number} index - The index of the message in the chat
 */

/**
 * @typedef {object} VectorItem
 * @property {string} type - Type of the item ('chat', 'file', 'world_info')
 * @property {string} text - The text content
 * @property {Object} metadata - Additional metadata for the item
 * @property {boolean} selected - Whether the item is selected for vectorization
 */

const MODULE_NAME = 'vectors-enhanced';

export const EXTENSION_PROMPT_TAG = '3_vectors';

const settings = {
  // Master switch - controls all plugin functionality
  master_enabled: true, // 主开关：控制整个插件的所有功能，默认启用

  // Vector source settings
  source: 'transformers',
  local_model: '', // 本地transformers模型名称
  vllm_model: '',
  vllm_url: '',
  ollama_model: 'rjmalagon/gte-qwen2-1.5b-instruct-embed-f16',
  ollama_url: '', // ollama API地址
  ollama_keep: false,

  // General vectorization settings
  auto_vectorize: true,
  chunk_size: 1000,
  overlap_percent: 10,
  score_threshold: 0.25,
  force_chunk_delimiter: '',

  // Query settings
  enabled: true, // 是否启用向量查询
  query_messages: 3, // 查询使用的最近消息数
  max_results: 10, // 返回的最大结果数

  // Injection settings
  template: '<must_know>以下是从相关背景知识库，包含重要的上下文、设定或细节：\n{{text}}</must_know>',
  position: extension_prompt_types.IN_PROMPT,
  depth: 2,
  depth_role: extension_prompt_roles.SYSTEM,
  include_wi: false,

  // Content tags
  content_tags: {
    chat: 'past_chat',
    file: 'databank',
    world_info: 'world_part',
  },

  // Content selection
  selected_content: {
    chat: {
      enabled: false,
      range: { start: 0, end: -1 },
      types: { user: true, assistant: true },
      tags: '', // comma-separated tag names to extract
      include_hidden: false, // 是否包含隐藏消息
    },
    files: { enabled: false, selected: [] },
    world_info: { enabled: false, selected: {} }, // { worldId: [entryIds] }
  },

  // Content filtering
  content_blacklist: [], // Array of keywords to filter out content

  // Vector tasks management
  vector_tasks: {}, // { chatId: [{ taskId, name, timestamp, settings, enabled }] }
};

const moduleWorker = new ModuleWorkerWrapper(synchronizeChat);
const cachedVectors = new Map(); // Cache for vectorized content
let syncBlocked = false;

/**
 * Generates a unique task ID
 * @returns {string} Unique task ID
 */
function generateTaskId() {
  return `task_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Gets all vector tasks for a chat
 * @param {string} chatId Chat ID
 * @returns {Array} Array of tasks
 */
function getChatTasks(chatId) {
  if (!settings.vector_tasks[chatId]) {
    settings.vector_tasks[chatId] = [];
  }
  return settings.vector_tasks[chatId];
}

/**
 * Adds a new vector task
 * @param {string} chatId Chat ID
 * @param {object} task Task object
 */
function addVectorTask(chatId, task) {
  const tasks = getChatTasks(chatId);
  tasks.push(task);
  settings.vector_tasks[chatId] = tasks;
  Object.assign(extension_settings.vectors_enhanced, settings);
  saveSettingsDebounced();
}

/**
 * Removes a vector task
 * @param {string} chatId Chat ID
 * @param {string} taskId Task ID to remove
 */
async function removeVectorTask(chatId, taskId) {
  const tasks = getChatTasks(chatId);
  const index = tasks.findIndex(t => t.taskId === taskId);
  if (index !== -1) {
    // Delete the vector collection
    await purgeVectorIndex(`${chatId}_${taskId}`);
    // Remove from tasks list
    tasks.splice(index, 1);
    settings.vector_tasks[chatId] = tasks;
    Object.assign(extension_settings.vectors_enhanced, settings);
    saveSettingsDebounced();
  }
}

/**
 * Gets the Collection ID for a file embedded in the chat.
 * @param {string} fileUrl URL of the file
 * @returns {string} Collection ID
 */
function getFileCollectionId(fileUrl) {
  return `file_${getHashValue(fileUrl)}`;
}

/**
 * Gets the chunk delimiters for splitting text.
 * @returns {string[]} Array of chunk delimiters
 */
function getChunkDelimiters() {
  const delimiters = ['\n\n', '\n', ' ', ''];
  if (settings.force_chunk_delimiter) {
    delimiters.unshift(settings.force_chunk_delimiter);
  }
  return delimiters;
}

/**
 * Splits text into chunks with optional overlap.
 * @param {string} text Text to split
 * @param {number} chunkSize Size of each chunk
 * @param {number} overlapPercent Overlap percentage
 * @returns {string[]} Array of text chunks
 */
function splitTextIntoChunks(text, chunkSize, overlapPercent) {
  const delimiters = getChunkDelimiters();
  const overlapSize = Math.round((chunkSize * overlapPercent) / 100);
  const adjustedChunkSize = overlapSize > 0 ? chunkSize - overlapSize : chunkSize;

  const chunks = splitRecursive(text, adjustedChunkSize, delimiters);

  if (overlapSize > 0) {
    return chunks.map((chunk, index) => overlapChunks(chunk, index, chunks, overlapSize));
  }

  return chunks;
}

/**
 * Modifies text chunks to include overlap with adjacent chunks.
 * @param {string} chunk Current item
 * @param {number} index Current index
 * @param {string[]} chunks List of chunks
 * @param {number} overlapSize Size of the overlap
 * @returns {string} Overlapped chunks
 */
function overlapChunks(chunk, index, chunks, overlapSize) {
  const halfOverlap = Math.floor(overlapSize / 2);
  const nextChunk = chunks[index + 1];
  const prevChunk = chunks[index - 1];

  const nextOverlap = trimToEndSentence(nextChunk?.substring(0, halfOverlap)) || '';
  const prevOverlap = trimToStartSentence(prevChunk?.substring(prevChunk.length - halfOverlap)) || '';
  const overlappedChunk = [prevOverlap, chunk, nextOverlap].filter(x => x).join(' ');

  return overlappedChunk;
}

/**
 * Parses tag configuration with exclusion syntax
 * @param {string} tagConfig Tag configuration string
 * @returns {object} Object with mainTag and excludeTags
 */
function parseTagWithExclusions(tagConfig) {
  if (!tagConfig.includes(' - ')) {
    return { mainTag: tagConfig, excludeTags: [] };
  }

  const [mainTag, excludePart] = tagConfig.split(' - ');
  const excludeTags = excludePart
    .split(',')
    .map(t => t.trim())
    .filter(t => t)
    .map(tag => {
      // 检测正则表达式格式：/pattern/flags
      const regexMatch = tag.match(/^\/(.+)\/([gimuy]*)$/);
      if (regexMatch) {
        return {
          type: 'regex',
          pattern: regexMatch[1],
          flags: regexMatch[2] || 'gi',
        };
      }
      // 传统标签格式
      return {
        type: 'tag',
        name: tag,
      };
    });

  return {
    mainTag: mainTag.trim(),
    excludeTags: excludeTags,
  };
}

/**
 * Removes excluded tags from content
 * @param {string} content Content to process
 * @param {string[]} excludeTags Tags to exclude
 * @returns {string} Content with excluded tags removed
 */
function removeExcludedTags(content, excludeTags) {
  let result = content;

  for (const excludeItem of excludeTags) {
    try {
      if (excludeItem.type === 'regex') {
        // 正则表达式排除
        const regex = new RegExp(excludeItem.pattern, excludeItem.flags);
        result = result.replace(regex, '');
      } else {
        // 传统标签排除
        const tagName = excludeItem.name;
        if (tagName.includes('<') && tagName.includes('>')) {
          const tagMatch = tagName.match(/<(\w+)(?:\s[^>]*)?>/);
          if (tagMatch) {
            const name = tagMatch[1];
            const regex = new RegExp(`<${name}(?:\\s[^>]*)?>[\\s\\S]*?<\\/${name}>`, 'gi');
            result = result.replace(regex, '');
          }
        } else {
          const regex = new RegExp(`<${tagName}(?:\\s[^>]*)?>[\\s\\S]*?<\\/${tagName}>`, 'gi');
          result = result.replace(regex, '');
        }
      }
    } catch (error) {
      console.warn(`标签排除错误: ${JSON.stringify(excludeItem)}`, error);
    }
  }

  return result
    .replace(/\n\s*\n\s*\n/g, '\n\n')
    .replace(/^\s+|\s+$/g, '')
    .replace(/\n\s+/g, '\n')
    .replace(/\s+\n/g, '\n')
    .trim();
}

/**
 * Checks if content should be skipped based on blacklist
 * @param {string} text Content to check
 * @param {string[]} blacklist Array of blacklist keywords
 * @returns {boolean} True if content should be skipped
 */
function shouldSkipContent(text, blacklist) {
  if (!blacklist || blacklist.length === 0) return false;

  const lowerText = text.toLowerCase();
  return blacklist.some(keyword => {
    const lowerKeyword = keyword.trim().toLowerCase();
    return lowerKeyword && lowerText.includes(lowerKeyword);
  });
}

/**
 * Extracts content from specific tags in a message
 * @param {string} text Message text
 * @param {string[]} tags Tags to extract (supports exclusion syntax)
 * @returns {string} Extracted content or original text if no tags specified
 */
function extractTagContent(text, tags) {
  if (!tags || tags.length === 0) return text;

  let extractedContent = [];
  const blacklist = settings.content_blacklist || [];

  for (const tagConfig of tags) {
    try {
      const { mainTag, excludeTags } = parseTagWithExclusions(tagConfig);

      // 第一步：提取主标签内容
      let mainContent = [];

      if (mainTag.includes(',')) {
        // 复杂标签配置：<details><summary>摘要</summary>,</details>
        const complexContent = extractComplexTag(text, mainTag);
        mainContent.push(...complexContent);
      } else if (mainTag.includes('<') && mainTag.includes('>')) {
        // HTML格式的简单标签：<content></content>
        const simpleContent = extractHtmlFormatTag(text, mainTag);
        mainContent.push(...simpleContent);
      } else {
        // 原始简单标签：content, thinking
        const simpleContent = extractSimpleTag(text, mainTag);
        mainContent.push(...simpleContent);
      }

      // 第二步：嵌套标签排除
      if (excludeTags.length > 0) {
        mainContent = mainContent
          .map(content => removeExcludedTags(content, excludeTags))
          .filter(content => content.trim()); // 移除空内容
      }

      // 第三步：黑名单过滤
      mainContent = mainContent.filter(content => {
        if (shouldSkipContent(content, blacklist)) {
          console.debug(`黑名单过滤跳过内容: ${content.substring(0, 50)}...`);
          return false;
        }
        return true;
      });

      extractedContent.push(...mainContent);
    } catch (error) {
      console.warn(`标签配置错误: ${tagConfig}`, error);
      // 继续处理其他标签，不因为一个错误而中断
    }
  }

  return extractedContent.length > 0 ? extractedContent.join('\n\n') : text;
}

/**
 * Extracts content using complex tag configuration
 * @param {string} text Text to search in
 * @param {string} tag Complex tag configuration like "<details><summary>摘要</summary>,</details>"
 * @returns {string[]} Array of extracted content
 */
function extractComplexTag(text, tag) {
  const parts = tag.split(',');
  if (parts.length !== 2) {
    throw new Error(`复杂标签配置格式错误，应该包含一个逗号: ${tag}`);
  }

  const startPattern = parts[0].trim(); // "<details><summary>摘要</summary>"
  const endPattern = parts[1].trim(); // "</details>"

  // 提取结束标签名
  const endTagMatch = endPattern.match(/<\/(\w+)>/);
  if (!endTagMatch) {
    throw new Error(`无法解析结束标签: ${endPattern}`);
  }
  const endTagName = endTagMatch[1]; // "details"

  // 构建匹配正则，提取中间内容
  const regex = new RegExp(`${escapeRegex(startPattern)}([\\s\\S]*?)<\\/${endTagName}>`, 'gi');

  const extractedContent = [];
  const matches = [...text.matchAll(regex)];

  matches.forEach(match => {
    if (match[1]) {
      // 提取中间的所有内容，包括HTML标签
      extractedContent.push(match[1].trim());
    }
  });

  return extractedContent;
}

/**
 * Extracts content using HTML format tag
 * @param {string} text Text to search in
 * @param {string} tag HTML format tag like "<content></content>"
 * @returns {string[]} Array of extracted content
 */
function extractHtmlFormatTag(text, tag) {
  // 提取标签名，处理可能的属性
  const tagMatch = tag.match(/<(\w+)(?:\s[^>]*)?>/);
  if (!tagMatch) {
    throw new Error(`无法解析HTML格式标签: ${tag}`);
  }
  const tagName = tagMatch[1];

  const extractedContent = [];
  const regex = new RegExp(`<${tagName}(?:\\s[^>]*)?>([\\s\\S]*?)<\\/${tagName}>`, 'gi');
  const matches = [...text.matchAll(regex)];

  matches.forEach(match => {
    if (match[1]) {
      extractedContent.push(match[1].trim());
    }
  });

  // 检查是否有未闭合的标签
  const openTags = (text.match(new RegExp(`<${tagName}(?:\\s[^>]*)?>`, 'gi')) || []).length;
  const closeTags = (text.match(new RegExp(`<\\/${tagName}>`, 'gi')) || []).length;

  if (openTags > closeTags) {
    console.warn(`警告: 发现 ${openTags - closeTags} 个未闭合的 <${tagName}> 标签`);
  }

  return extractedContent;
}

/**
 * Extracts content using simple tag name
 * @param {string} text Text to search in
 * @param {string} tag Simple tag name like "content" or "thinking"
 * @returns {string[]} Array of extracted content
 */
function extractSimpleTag(text, tag) {
  const extractedContent = [];
  const regex = new RegExp(`<${tag}>([\\s\\S]*?)<\\/${tag}>`, 'gi');
  const matches = [...text.matchAll(regex)];

  matches.forEach(match => {
    if (match[1]) {
      extractedContent.push(match[1].trim());
    }
  });

  // 检查是否有未闭合的标签
  const openTags = (text.match(new RegExp(`<${tag}>`, 'gi')) || []).length;
  const closeTags = (text.match(new RegExp(`<\\/${tag}>`, 'gi')) || []).length;

  if (openTags > closeTags) {
    console.warn(`警告: 发现 ${openTags - closeTags} 个未闭合的 <${tag}> 标签`);
  }

  return extractedContent;
}

/**
 * Escapes special regex characters in a string
 * @param {string} str String to escape
 * @returns {string} Escaped string
 */
function escapeRegex(str) {
  return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * Gets all vectorizable content based on current settings
 * @returns {Promise<VectorItem[]>} Array of vector items
 */
async function getVectorizableContent() {
  const items = [];
  const context = getContext();

  // Chat messages
  if (settings.selected_content.chat.enabled && context.chat) {
    const chatSettings = settings.selected_content.chat;
    const start = chatSettings.range?.start || 0;
    const end = chatSettings.range?.end || -1;
    const types = chatSettings.types || { user: true, assistant: true };
    const tags = chatSettings.tags || '';
    const blacklist = settings.content_blacklist || [];

    const messages = context.chat.slice(start, end === -1 ? undefined : end);
    const tagList = tags
      ? tags
          .split(',')
          .map(t => t.trim())
          .filter(t => t)
      : [];

    messages.forEach((msg, idx) => {
      // 处理隐藏消息
      if (msg.is_system === true && !chatSettings.include_hidden) {
        return; // 跳过隐藏的消息（除非明确要包含）
      }

      if (!types.user && msg.is_user) return;
      if (!types.assistant && !msg.is_user) return;

      const extractedText = extractTagContent(substituteParams(msg.mes), tagList);

      items.push({
        type: 'chat',
        text: extractedText,
        metadata: {
          index: start + idx,
          is_user: msg.is_user,
          name: msg.name,
          is_hidden: msg.is_system === true,
        },
        selected: true,
      });
    });
  }

  // Files
  if (settings.selected_content.files.enabled) {
    const allFiles = [
      ...getDataBankAttachments(),
      ...getDataBankAttachmentsForSource('global'),
      ...getDataBankAttachmentsForSource('character'),
      ...getDataBankAttachmentsForSource('chat'),
      ...context.chat.filter(x => x.extra?.file).map(x => x.extra.file),
    ];

    for (const file of allFiles) {
      if (!settings.selected_content.files.selected.includes(file.url)) continue;

      const text = await getFileAttachment(file.url);
      items.push({
        type: 'file',
        text: text,
        metadata: {
          name: file.name,
          url: file.url,
          size: file.size,
        },
        selected: true,
      });
    }
  }

  // World Info
  if (settings.selected_content.world_info.enabled) {
    const entries = await getSortedEntries();

    for (const entry of entries) {
      if (!entry.world || !entry.content || entry.disable) continue;

      const selectedEntries = settings.selected_content.world_info.selected[entry.world] || [];
      if (!selectedEntries.includes(entry.uid)) continue;

      items.push({
        type: 'world_info',
        text: entry.content,
        metadata: {
          world: entry.world,
          uid: entry.uid,
          key: entry.key.join(', '),
          comment: entry.comment,
        },
        selected: true,
      });
    }
  }

  return items;
}

/**
 * Updates progress display
 * @param {number} current Current progress
 * @param {number} total Total items
 * @param {string} message Progress message
 */
function updateProgress(current, total, message) {
  const percent = Math.round((current / total) * 100);
  $('#vectors_enhanced_progress').show();
  $('#vectors_enhanced_progress .progress-bar-inner').css('width', `${percent}%`);
  $('#vectors_enhanced_progress .progress-text').text(`${message} (${current}/${total})`);
}

/**
 * Hides progress display
 */
function hideProgress() {
  $('#vectors_enhanced_progress').hide();
  $('#vectors_enhanced_progress .progress-bar-inner').css('width', '0%');
  $('#vectors_enhanced_progress .progress-text').text('准备中...');
}

/**
 * Generates a task name based on settings
 * @param {object} chatSettings Chat settings
 * @param {number} itemCount Number of items
 * @returns {Promise<string>} Task name
 */
async function generateTaskName(chatSettings, itemCount) {
  const parts = [];

  // Chat range
  if (settings.selected_content.chat.enabled) {
    const start = chatSettings.range?.start || 0;
    const end = chatSettings.range?.end || -1;
    if (end === -1) {
      parts.push(`消息 #${start} 到最后`);
    } else {
      parts.push(`消息 #${start}-${end}`);
    }
  }

  // Files - only count if enabled
  if (settings.selected_content.files.enabled) {
    const fileCount = settings.selected_content.files.selected.length;
    if (fileCount > 0) {
      parts.push(`${fileCount} 个文件`);
    }
  }

  // World info - only count if enabled
  if (settings.selected_content.world_info.enabled) {
    const wiCount = Object.values(settings.selected_content.world_info.selected).flat().length;
    if (wiCount > 0) {
      parts.push(`${wiCount} 条世界信息`);
    }
  }

  // If no specific content selected, use generic name
  if (parts.length === 0) {
    parts.push(`${itemCount} 个项目`);
  }

  // Add timestamp
  const time = new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
  return `${parts.join(', ')} (${time})`;
}

/**
 * Updates the task list UI
 */
async function updateTaskList() {
  const chatId = getCurrentChatId();
  if (!chatId) return;

  const tasks = getChatTasks(chatId);
  const taskList = $('#vectors_enhanced_task_list');
  taskList.empty();

  if (tasks.length === 0) {
    taskList.append('<div class="text-muted">没有向量化任务</div>');
    return;
  }

  tasks.forEach((task, index) => {
    const taskDiv = $('<div class="vector-enhanced-task-item"></div>');

    const checkbox = $(`
            <label class="checkbox_label flex-container alignItemsCenter">
                <input type="checkbox" ${task.enabled ? 'checked' : ''} />
                <span class="flex1">
                    <strong>${task.name}</strong>
                    <small class="text-muted"> - ${new Date(task.timestamp).toLocaleString('zh-CN')}</small>
                </span>
            </label>
        `);

    checkbox.find('input').on('change', function () {
      task.enabled = this.checked;
      Object.assign(extension_settings.vectors_enhanced, settings);
      saveSettingsDebounced();
    });

    const deleteBtn = $(`<button class="menu_button menu_button_icon" title="删除此任务">
            <i class="fa-solid fa-trash"></i>
        </button>`);

    deleteBtn.on('click', async () => {
      const confirm = await callGenericPopup('确定要删除这个向量化任务吗？', POPUP_TYPE.CONFIRM);
      if (confirm === POPUP_RESULT.AFFIRMATIVE) {
        await removeVectorTask(chatId, task.taskId);
        await updateTaskList();
        toastr.success('任务已删除');
      }
    });

    taskDiv.append(checkbox);
    taskDiv.append(deleteBtn);
    taskList.append(taskDiv);
  });
}

/**
 * Vectorizes selected content
 * @returns {Promise<void>}
 */
async function vectorizeContent() {
  const items = await getVectorizableContent();
  if (items.length === 0) {
    toastr.warning('未选择要向量化的内容');
    return;
  }

  const chatId = getCurrentChatId();
  if (!chatId) {
    toastr.error('未选择聊天');
    return;
  }

  // Generate task name
  const context = getContext();
  const chatSettings = settings.selected_content.chat;
  const taskName = await generateTaskName(chatSettings, items.length);

  try {
    toastr.info('向量化开始...', '处理中');
    updateProgress(0, items.length, '准备向量化');

    // Create new task
    const taskId = generateTaskId();
    const task = {
      taskId: taskId,
      name: taskName,
      timestamp: Date.now(),
      settings: JSON.parse(JSON.stringify(settings.selected_content)),
      enabled: true,
      itemCount: items.length,
    };

    // Use task-specific collection ID
    const collectionId = `${chatId}_${taskId}`;

    // Process items in chunks
    const allChunks = [];
    let processedItems = 0;

    for (const item of items) {
      const chunks = splitTextIntoChunks(item.text, settings.chunk_size, settings.overlap_percent);
      chunks.forEach((chunk, idx) => {
        allChunks.push({
          hash: getHashValue(chunk),
          text: chunk,
          index: allChunks.length,
          metadata: {
            ...item.metadata,
            type: item.type,
            chunk_index: idx,
            chunk_total: chunks.length,
          },
        });
      });

      processedItems++;
      updateProgress(processedItems, items.length, '正在处理内容');
    }

    // Insert vectors in batches
    updateProgress(0, allChunks.length, '正在插入向量');
    const batchSize = 50;
    for (let i = 0; i < allChunks.length; i += batchSize) {
      const batch = allChunks.slice(i, Math.min(i + batchSize, allChunks.length));
      await insertVectorItems(collectionId, batch);
      updateProgress(Math.min(i + batchSize, allChunks.length), allChunks.length, '正在插入向量');
    }

    // Add task to list
    addVectorTask(chatId, task);

    // Update cache for this task
    cachedVectors.set(collectionId, {
      timestamp: Date.now(),
      items: allChunks,
      settings: JSON.parse(JSON.stringify(settings)),
    });

    hideProgress();
    toastr.success(`成功创建向量化任务 "${taskName}"：${items.length} 个项目，${allChunks.length} 个块`, '成功');

    // Refresh task list UI
    await updateTaskList();
  } catch (error) {
    console.error('向量化失败:', error);
    hideProgress();
    toastr.error('向量化内容失败', '错误');
  }
}

/**
 * Exports vectorized content
 * @returns {Promise<void>}
 */
async function exportVectors() {
  const context = getContext();
  const chatId = getCurrentChatId();

  if (!chatId) {
    toastr.error('未选择聊天');
    return;
  }

  const items = await getVectorizableContent();
  if (items.length === 0) {
    toastr.warning('未选择要导出的内容');
    return;
  }

  // Build export content
  let exportText = `角色卡：${context.name || '未知'}\n`;
  exportText += `时间：${new Date().toLocaleString('zh-CN')}\n\n`;

  // Group items by type
  const grouped = items.reduce((acc, item) => {
    if (!acc[item.type]) acc[item.type] = [];
    acc[item.type].push(item);
    return acc;
  }, {});

  // Files
  exportText += '=== 数据库文件 ===\n';
  if (grouped.file && grouped.file.length > 0) {
    grouped.file.forEach(item => {
      exportText += `文件名：${item.metadata.name}\n`;
      exportText += `内容：\n${item.text}\n\n`;
    });
  } else {
    exportText += '无\n\n';
  }

  // World Info
  exportText += '=== 世界书 ===\n';
  if (grouped.world_info && grouped.world_info.length > 0) {
    grouped.world_info.forEach(item => {
      exportText += `世界：${item.metadata.world}\n`;
      exportText += `注释：${item.metadata.comment || '无'}\n`;
      exportText += `内容：${item.text}\n\n`;
    });
  } else {
    exportText += '无\n\n';
  }

  // Chat messages
  exportText += '=== 聊天记录 ===\n';
  if (grouped.chat && grouped.chat.length > 0) {
    grouped.chat.forEach(item => {
      exportText += `#${item.metadata.index}：${item.text}\n\n`;
    });
  } else {
    exportText += '无\n\n';
  }

  // Create and download file
  const blob = new Blob([exportText], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `向量导出_${context.name || chatId}_${Date.now()}.txt`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);

  toastr.success('导出成功');
}

/**
 * Previews vectorizable content
 * @returns {Promise<void>}
 */
async function previewContent() {
  const items = await getVectorizableContent();
  if (items.length === 0) {
    toastr.warning('未选择要预览的内容');
    return;
  }

  // 统计过滤信息
  let totalOriginalBlocks = 0;
  let excludedBlocks = 0;
  let blacklistedBlocks = 0;
  let finalBlocks = 0;

  // 模拟处理过程以获取统计信息
  const blacklist = settings.content_blacklist || [];
  const chatSettings = settings.selected_content.chat;
  const tags = chatSettings.tags
    ? chatSettings.tags
        .split(',')
        .map(t => t.trim())
        .filter(t => t)
    : [];

  if (chatSettings.enabled && tags.length > 0) {
    const context = getContext();
    const start = chatSettings.range?.start || 0;
    const end = chatSettings.range?.end || -1;
    const messages = context.chat.slice(start, end === -1 ? undefined : end);

    messages.forEach(msg => {
      if (msg.is_system) return;
      if (!chatSettings.types.user && msg.is_user) return;
      if (!chatSettings.types.assistant && !msg.is_user) return;

      const originalText = substituteParams(msg.mes);

      for (const tagConfig of tags) {
        try {
          const { mainTag, excludeTags } = parseTagWithExclusions(tagConfig);

          // 统计原始提取的块
          let originalBlocks = [];
          if (mainTag.includes(',')) {
            originalBlocks = extractComplexTag(originalText, mainTag);
          } else if (mainTag.includes('<') && mainTag.includes('>')) {
            originalBlocks = extractHtmlFormatTag(originalText, mainTag);
          } else {
            originalBlocks = extractSimpleTag(originalText, mainTag);
          }

          totalOriginalBlocks += originalBlocks.length;

          // 统计排除的块
          let afterExclusion = originalBlocks;
          if (excludeTags.length > 0) {
            afterExclusion = originalBlocks
              .map(content => removeExcludedTags(content, excludeTags))
              .filter(content => content.trim());
            excludedBlocks += originalBlocks.length - afterExclusion.length;
          }

          // 统计黑名单过滤的块
          const afterBlacklist = afterExclusion.filter(content => {
            if (shouldSkipContent(content, blacklist)) {
              blacklistedBlocks++;
              return false;
            }
            return true;
          });

          finalBlocks += afterBlacklist.length;
        } catch (error) {
          console.warn(`统计时标签配置错误: ${tagConfig}`, error);
        }
      }
    });
  }

  let html = '<div class="vector-preview">';
  html += `<div class="preview-header">已选择内容（${items.length} 项）</div>`;

  // 添加过滤统计信息
  if (totalOriginalBlocks > 0) {
    html +=
      '<div class="preview-stats" style="background: var(--SmartThemeQuoteColor); padding: 0.5rem; margin-bottom: 1rem; border-radius: 4px;">';
    html += '<div style="font-weight: bold; margin-bottom: 0.25rem;">内容处理统计：</div>';
    html += `<div>• 原始提取块数：${totalOriginalBlocks}</div>`;
    if (excludedBlocks > 0) {
      html += `<div>• 嵌套标签排除：${excludedBlocks} 个块被移除</div>`;
    }
    if (blacklistedBlocks > 0) {
      html += `<div>• 黑名单过滤：${blacklistedBlocks} 个块被跳过</div>`;
    }
    html += `<div>• 最终向量化：${finalBlocks} 个块</div>`;
    html += '</div>';
  }

  html += '<div class="preview-sections">';

  // Group by type
  const grouped = items.reduce((acc, item) => {
    if (!acc[item.type]) acc[item.type] = [];
    acc[item.type].push(item);
    return acc;
  }, {});

  // Files section
  html += '<div class="preview-section">';
  html += `<div class="preview-section-title">文件（${grouped.file?.length || 0}）</div>`;
  html += '<div class="preview-section-content">';
  if (grouped.file && grouped.file.length > 0) {
    grouped.file.forEach(item => {
      const sizeKB = (item.metadata.size / 1024).toFixed(1);
      html += `<div class="preview-item">`;
      html += `<strong>${item.metadata.name}</strong> - ${sizeKB} KB`;
      html += `</div>`;
    });
  } else {
    html += '<div class="preview-empty">无文件</div>';
  }
  html += '</div></div>';

  // World Info section
  html += '<div class="preview-section">';
  html += `<div class="preview-section-title">世界信息（${grouped.world_info?.length || 0}）</div>`;
  html += '<div class="preview-section-content">';
  if (grouped.world_info && grouped.world_info.length > 0) {
    // Group by world
    const byWorld = {};
    grouped.world_info.forEach(item => {
      if (!byWorld[item.metadata.world]) byWorld[item.metadata.world] = [];
      byWorld[item.metadata.world].push(item);
    });

    for (const [world, entries] of Object.entries(byWorld)) {
      html += `<div class="preview-world-group">`;
      html += `<div class="preview-world-name">${world}</div>`;
      entries.forEach(entry => {
        html += `<div class="preview-world-entry">${entry.metadata.comment || '(无注释)'}</div>`;
      });
      html += `</div>`;
    }
  } else {
    html += '<div class="preview-empty">无世界信息</div>';
  }
  html += '</div></div>';

  // Chat messages section
  html += '<div class="preview-section">';
  html += `<div class="preview-section-title">聊天记录（${grouped.chat?.length || 0} 条消息）</div>`;
  html += '<div class="preview-section-content">';
  if (grouped.chat && grouped.chat.length > 0) {
    grouped.chat.forEach(item => {
      const msgType = item.metadata.is_user ? '用户' : 'AI';
      html += `<div class="preview-chat-message">`;
      html += `<div class="preview-chat-header">#${item.metadata.index} - ${msgType}（${item.metadata.name}）</div>`;
      html += `<div class="preview-chat-content">${item.text}</div>`;
      html += `</div>`;
    });
  } else {
    html += '<div class="preview-empty">无聊天记录</div>';
  }
  html += '</div></div>';

  html += '</div></div>';

  await callGenericPopup(html, POPUP_TYPE.TEXT, '', {
    okButton: '关闭',
    wide: true,
    large: true,
  });
}

/**
 * Cache object for storing hash values
 * @type {Map<string, number>}
 */
const hashCache = new Map();

/**
 * Gets the hash value for a given string
 * @param {string} str Input string
 * @returns {number} Hash value
 */
function getHashValue(str) {
  if (hashCache.has(str)) {
    return hashCache.get(str);
  }
  const hash = getStringHash(str);
  hashCache.set(str, hash);
  return hash;
}

/**
 * Synchronizes chat vectors
 * @param {number} batchSize Batch size for processing
 * @returns {Promise<number>} Number of remaining items
 */
async function synchronizeChat(batchSize = 5) {
  // 检查主开关是否启用
  if (!settings.master_enabled) {
    return -1;
  }

  if (!settings.auto_vectorize) {
    return -1;
  }

  try {
    await waitUntilCondition(() => !syncBlocked && !is_send_press, 1000);
  } catch {
    console.log('Vectors: Synchronization blocked by another process');
    return -1;
  }

  try {
    syncBlocked = true;
    // Auto-vectorization logic will be implemented based on settings
    return -1;
  } finally {
    syncBlocked = false;
  }
}

/**
 * Retrieves vectorized content for injection
 * @param {object[]} chat Chat messages
 * @param {number} contextSize Context size
 * @param {function} abort Abort function
 * @param {string} type Generation type
 */
async function rearrangeChat(chat, contextSize, abort, type) {
  try {
    if (type === 'quiet') {
      console.debug('Vectors: Skipping quiet prompt');
      return;
    }

    setExtensionPrompt(
      EXTENSION_PROMPT_TAG,
      '',
      settings.position,
      settings.depth,
      settings.include_wi,
      settings.depth_role,
    );

    // 检查主开关是否启用
    if (!settings.master_enabled) {
      console.debug('Vectors: Master switch disabled, skipping all functionality');
      return;
    }

    // 检查是否启用向量查询
    if (!settings.enabled) {
      console.debug('Vectors: Query disabled by user');
      return;
    }

    const chatId = getCurrentChatId();
    if (!chatId) {
      console.debug('Vectors: No chat ID available');
      return;
    }

    // Query vectors based on recent messages
    const queryMessages = Math.min(settings.query_messages || 3, chat.length);
    const queryText = chat
      .slice(-queryMessages)
      .map(x => x.mes)
      .join('\n');
    if (!queryText.trim()) return;

    // Get all enabled tasks for this chat
    const tasks = getChatTasks(chatId).filter(t => t.enabled);
    if (tasks.length === 0) {
      console.debug('Vectors: No enabled tasks for this chat');
      return;
    }

    // Query all enabled tasks
    const allResults = [];
    for (const task of tasks) {
      const collectionId = `${chatId}_${task.taskId}`;
      try {
        const results = await queryCollection(collectionId, queryText, settings.max_results || 10);
        console.debug(`Vectors: Query results for task ${task.name}:`, results);

        // 根据API返回的结构处理结果
        if (results) {
          // 如果API返回了items数组（包含text）
          if (results.items && Array.isArray(results.items)) {
            results.items.forEach(item => {
              if (item.text) {
                allResults.push({
                  text: item.text,
                  score: item.score || 0,
                  metadata: {
                    ...item.metadata,
                    taskName: task.name,
                    taskId: task.taskId,
                  },
                });
              }
            });
          }
          // 如果API只返回了hashes和metadata，尝试从缓存获取
          else if (results.hashes && results.metadata) {
            const cachedData = cachedVectors.get(collectionId);
            if (cachedData && cachedData.items) {
              results.hashes.forEach((hash, index) => {
                const cachedItem = cachedData.items.find(item => item.hash === hash);
                if (cachedItem && cachedItem.text) {
                  allResults.push({
                    text: cachedItem.text,
                    score: results.metadata[index]?.score || 0,
                    metadata: {
                      ...cachedItem.metadata,
                      ...(results.metadata[index] || {}),
                      taskName: task.name,
                      taskId: task.taskId,
                    },
                  });
                }
              });
            } else {
              console.warn(`Vectors: No cached data for collection ${collectionId}, cannot retrieve text`);
            }
          }
        }
      } catch (error) {
        console.error(`Vectors: Failed to query task ${task.name}:`, error);
      }
    }

    if (allResults.length === 0) {
      console.debug('Vectors: No query results found');
      return;
    }

    console.debug(`Vectors: Found ${allResults.length} total results`);

    // Sort by score and take top results
    allResults.sort((a, b) => (b.score || 0) - (a.score || 0));
    const topResults = allResults.slice(0, settings.max_results || 10);

    console.debug(`Vectors: Using top ${topResults.length} results`);

    // Group results by type
    const groupedResults = {};
    topResults.forEach(result => {
      const type = result.metadata?.type || 'unknown';
      if (!groupedResults[type]) {
        groupedResults[type] = [];
      }
      groupedResults[type].push(result);
    });

    console.debug(
      'Vectors: Grouped results by type:',
      Object.keys(groupedResults).map(k => `${k}: ${groupedResults[k].length}`),
    );

    // Format results with tags
    const formattedParts = [];

    // Process chat messages
    if (groupedResults.chat && groupedResults.chat.length > 0) {
      const chatTexts = groupedResults.chat
        .sort((a, b) => (a.metadata?.index || 0) - (b.metadata?.index || 0))
        .map(m => m.text)
        .filter(onlyUnique)
        .join('\n\n');

      const tag = settings.content_tags?.chat || 'past_chat';
      formattedParts.push(`<${tag}>\n${chatTexts}\n</${tag}>`);
    }

    // Process world info
    if (groupedResults.world_info && groupedResults.world_info.length > 0) {
      const wiTexts = groupedResults.world_info
        .map(m => m.text)
        .filter(onlyUnique)
        .join('\n\n');

      const tag = settings.content_tags?.world_info || 'world_part';
      formattedParts.push(`<${tag}>\n${wiTexts}\n</${tag}>`);
    }

    // Process files
    if (groupedResults.file && groupedResults.file.length > 0) {
      const fileTexts = groupedResults.file
        .map(m => m.text)
        .filter(onlyUnique)
        .join('\n\n');

      const tag = settings.content_tags?.file || 'databank';
      formattedParts.push(`<${tag}>\n${fileTexts}\n</${tag}>`);
    }

    // Join all parts
    const relevantTexts = formattedParts.join('\n\n');

    console.debug(`Vectors: Formatted ${formattedParts.length} parts, total length: ${relevantTexts.length}`);

    if (!relevantTexts) {
      console.debug('Vectors: No relevant texts found after formatting');
      return;
    }

    const insertedText = substituteParamsExtended(settings.template, { text: relevantTexts });
    console.debug(`Vectors: Final injected text length: ${insertedText.length}`);

    setExtensionPrompt(
      EXTENSION_PROMPT_TAG,
      insertedText,
      settings.position,
      settings.depth,
      settings.include_wi,
      settings.depth_role,
    );
  } catch (error) {
    console.error('Vectors: Failed to rearrange chat', error);
  }
}

window['vectors_rearrangeChat'] = rearrangeChat;

// 全局事件绑定 - 确保按钮始终有效
$(document).on('click', '#vectors_enhanced_preview', async function (e) {
  e.preventDefault();
  console.log('预览按钮被点击 (全局绑定)');

  if (!settings.master_enabled) {
    toastr.warning('请先启用聊天记录超级管理器');
    return;
  }

  try {
    await previewContent();
  } catch (error) {
    console.error('预览错误:', error);
    toastr.error('预览失败: ' + error.message);
  }
});

$(document).on('click', '#vectors_enhanced_export', async function (e) {
  e.preventDefault();
  console.log('导出按钮被点击 (全局绑定)');

  if (!settings.master_enabled) {
    toastr.warning('请先启用聊天记录超级管理器');
    return;
  }

  try {
    await exportVectors();
  } catch (error) {
    console.error('导出错误:', error);
    toastr.error('导出失败: ' + error.message);
  }
});

$(document).on('click', '#vectors_enhanced_vectorize', async function (e) {
  e.preventDefault();
  console.log('向量化按钮被点击 (全局绑定)');

  if (!settings.master_enabled) {
    toastr.warning('请先启用聊天记录超级管理器');
    return;
  }

  try {
    await vectorizeContent();
  } catch (error) {
    console.error('向量化错误:', error);
    toastr.error('向量化失败: ' + error.message);
  }
});

/**
 * Gets request body for vector operations
 * @param {object} args Additional arguments
 * @returns {object} Request body
 */
function getVectorsRequestBody(args = {}) {
  const body = Object.assign({}, args);

  switch (settings.source) {
    case 'transformers':
      // Local transformers
      if (settings.local_model) {
        body.model = settings.local_model;
      }
      break;
    case 'vllm':
      body.apiUrl = settings.vllm_url || textgenerationwebui_settings.server_urls[textgen_types.VLLM];
      body.model = settings.vllm_model;
      break;
    case 'ollama':
      body.model = settings.ollama_model;
      body.apiUrl =
        settings.ollama_url ||
        textgenerationwebui_settings.server_urls[textgen_types.OLLAMA] ||
        'http://localhost:11434';
      body.keep = !!settings.ollama_keep;
      break;
  }

  body.source = settings.source;
  return body;
}

/**
 * Throws if the vector source is invalid
 */
function throwIfSourceInvalid() {
  if (settings.source === 'vllm') {
    if (!settings.vllm_url && !textgenerationwebui_settings.server_urls[textgen_types.VLLM]) {
      throw new Error('vLLM URL not configured');
    }
    if (!settings.vllm_model) {
      throw new Error('vLLM model not specified');
    }
  }

  if (settings.source === 'ollama') {
    if (!settings.ollama_url && !textgenerationwebui_settings.server_urls[textgen_types.OLLAMA]) {
      throw new Error('Ollama URL not configured');
    }
    if (!settings.ollama_model) {
      throw new Error('Ollama model not specified');
    }
    // ollama_url 是可选的，因为有默认值 http://localhost:11434
  }
}

/**
 * Gets saved hashes for a collection
 * @param {string} collectionId Collection ID
 * @returns {Promise<number[]>} Array of hashes
 */
async function getSavedHashes(collectionId) {
  const response = await fetch('/api/vector/list', {
    method: 'POST',
    headers: getRequestHeaders(),
    body: JSON.stringify({
      ...getVectorsRequestBody(),
      collectionId: collectionId,
    }),
  });

  if (!response.ok) {
    throw new Error(`Failed to get saved hashes for collection ${collectionId}`);
  }

  return await response.json();
}

/**
 * Inserts vector items into a collection
 * @param {string} collectionId Collection ID
 * @param {object[]} items Items to insert
 * @returns {Promise<void>}
 */
async function insertVectorItems(collectionId, items) {
  throwIfSourceInvalid();

  const response = await fetch('/api/vector/insert', {
    method: 'POST',
    headers: getRequestHeaders(),
    body: JSON.stringify({
      ...getVectorsRequestBody(),
      collectionId: collectionId,
      items: items,
    }),
  });

  if (!response.ok) {
    throw new Error(`Failed to insert vector items for collection ${collectionId}`);
  }
}

/**
 * Queries a collection
 * @param {string} collectionId Collection ID
 * @param {string} searchText Search text
 * @param {number} topK Number of results
 * @returns {Promise<{hashes: number[], metadata: object[]}>}
 */
async function queryCollection(collectionId, searchText, topK) {
  const response = await fetch('/api/vector/query', {
    method: 'POST',
    headers: getRequestHeaders(),
    body: JSON.stringify({
      ...getVectorsRequestBody(),
      collectionId: collectionId,
      searchText: searchText,
      topK: topK,
      threshold: settings.score_threshold,
      includeText: true, // 请求包含文本内容
    }),
  });

  if (!response.ok) {
    throw new Error(`Failed to query collection ${collectionId}`);
  }

  const result = await response.json();
  console.debug(`Vectors: Raw query result for ${collectionId}:`, result);
  return result;
}

/**
 * Purges a vector index
 * @param {string} collectionId Collection ID
 * @returns {Promise<boolean>} Success status
 */
async function purgeVectorIndex(collectionId) {
  try {
    const response = await fetch('/api/vector/purge', {
      method: 'POST',
      headers: getRequestHeaders(),
      body: JSON.stringify({
        ...getVectorsRequestBody(),
        collectionId: collectionId,
      }),
    });

    if (!response.ok) {
      throw new Error(`Could not delete vector index for collection ${collectionId}`);
    }

    console.log(`Vectors: Purged vector index for collection ${collectionId}`);
    cachedVectors.delete(collectionId);
    return true;
  } catch (error) {
    console.error('Vectors: Failed to purge', error);
    return false;
  }
}

/**
 * Updates UI based on settings
 */
function toggleSettings() {
  $('#vectors_enhanced_vllm_settings').toggle(settings.source === 'vllm');
  $('#vectors_enhanced_ollama_settings').toggle(settings.source === 'ollama');
  $('#vectors_enhanced_local_settings').toggle(settings.source === 'transformers');
}

/**
 * Updates UI state based on master switch
 */
function updateMasterSwitchState() {
  const isEnabled = settings.master_enabled;

  // 控制主要设置区域的显示/隐藏
  $('#vectors_enhanced_main_settings').toggle(isEnabled);
  $('#vectors_enhanced_content_settings').toggle(isEnabled);
  $('#vectors_enhanced_tasks_settings').toggle(isEnabled);
  $('#vectors_enhanced_actions_settings').toggle(isEnabled);

  // 如果禁用，还需要禁用所有输入控件（作为额外保护）
  const settingsContainer = $('#vectors_enhanced_container');
  settingsContainer
    .find('input, select, textarea, button')
    .not('#vectors_enhanced_master_enabled')
    .prop('disabled', !isEnabled);

  // 更新视觉效果
  if (isEnabled) {
    settingsContainer.removeClass('vectors-disabled');
  } else {
    settingsContainer.addClass('vectors-disabled');
  }
}

/**
 * Updates content selection UI
 */
function updateContentSelection() {
  // This will be called when settings change to update the UI
  $('#vectors_enhanced_chat_settings').toggle(settings.selected_content.chat.enabled);
  $('#vectors_enhanced_files_settings').toggle(settings.selected_content.files.enabled);
  $('#vectors_enhanced_wi_settings').toggle(settings.selected_content.world_info.enabled);
}

/**
 * Updates the file list UI
 */
async function updateFileList() {
  const fileList = $('#vectors_enhanced_file_list');
  fileList.empty();

  const context = getContext();
  const allFiles = [
    ...getDataBankAttachments(),
    ...getDataBankAttachmentsForSource('global'),
    ...getDataBankAttachmentsForSource('character'),
    ...getDataBankAttachmentsForSource('chat'),
    ...context.chat.filter(x => x.extra?.file).map(x => x.extra.file),
  ];

  if (allFiles.length === 0) {
    fileList.append('<div class="text-muted">没有可用文件</div>');
    return;
  }

  // Group files by source
  const dataBankFiles = getDataBankAttachments();
  const chatFiles = context.chat?.filter(x => x.extra?.file).map(x => x.extra.file) || [];

  if (dataBankFiles.length > 0) {
    fileList.append('<div class="file-group-header">数据库文件</div>');
    dataBankFiles.forEach(file => {
      const isChecked = settings.selected_content.files.selected.includes(file.url);
      const checkbox = $(`
                <label class="checkbox_label flex-container alignItemsCenter" title="${file.name}">
                    <input type="checkbox" value="${file.url}" ${isChecked ? 'checked' : ''} />
                    <span class="flex1 text-overflow-ellipsis">${file.name} (${(file.size / 1024).toFixed(1)} KB)</span>
                </label>
            `);

      checkbox.find('input').on('change', function () {
        if (this.checked) {
          if (!settings.selected_content.files.selected.includes(file.url)) {
            settings.selected_content.files.selected.push(file.url);
          }
        } else {
          settings.selected_content.files.selected = settings.selected_content.files.selected.filter(
            url => url !== file.url,
          );
        }
        Object.assign(extension_settings.vectors_enhanced, settings);
        saveSettingsDebounced();
      });

      fileList.append(checkbox);
    });
  }

  if (chatFiles.length > 0) {
    if (dataBankFiles.length > 0) fileList.append('<hr class="m-t-0-5 m-b-0-5">');
    fileList.append('<div class="file-group-header">聊天附件</div>');
    chatFiles.forEach(file => {
      const isChecked = settings.selected_content.files.selected.includes(file.url);
      const checkbox = $(`
                <label class="checkbox_label flex-container alignItemsCenter" title="${file.name}">
                    <input type="checkbox" value="${file.url}" ${isChecked ? 'checked' : ''} />
                    <span class="flex1 text-overflow-ellipsis">${file.name} (${(file.size / 1024).toFixed(1)} KB)</span>
                </label>
            `);

      checkbox.find('input').on('change', function () {
        if (this.checked) {
          if (!settings.selected_content.files.selected.includes(file.url)) {
            settings.selected_content.files.selected.push(file.url);
          }
        } else {
          settings.selected_content.files.selected = settings.selected_content.files.selected.filter(
            url => url !== file.url,
          );
        }
        Object.assign(extension_settings.vectors_enhanced, settings);
        saveSettingsDebounced();
      });

      fileList.append(checkbox);
    });
  }
}

/**
 * Updates the World Info list UI
 */
async function updateWorldInfoList() {
  const entries = await getSortedEntries();
  const wiList = $('#vectors_enhanced_wi_list');
  wiList.empty();

  if (!entries || entries.length === 0) {
    wiList.append('<div class="text-muted">没有可用的世界信息条目</div>');
    return;
  }

  // Group entries by world
  const grouped = {};
  entries.forEach(entry => {
    if (!entry.world || entry.disable || !entry.content) return;
    if (!grouped[entry.world]) grouped[entry.world] = [];
    grouped[entry.world].push(entry);
  });

  if (Object.keys(grouped).length === 0) {
    wiList.append('<div class="text-muted">未找到有效的世界信息条目</div>');
    return;
  }

  for (const [world, worldEntries] of Object.entries(grouped)) {
    const worldDiv = $('<div class="wi-world-group"></div>');

    // 世界名称和全选复选框
    const selectedEntries = settings.selected_content.world_info.selected[world] || [];
    const allChecked = worldEntries.length > 0 && worldEntries.every(e => selectedEntries.includes(e.uid));

    const worldHeader = $(`
            <div class="wi-world-header flex-container alignItemsCenter">
                <label class="checkbox_label flex1">
                    <input type="checkbox" class="world-select-all" data-world="${world}" ${
      allChecked ? 'checked' : ''
    } />
                    <span class="wi-world-name">${world}</span>
                </label>
            </div>
        `);

    // 全选复选框事件
    worldHeader.find('.world-select-all').on('change', function () {
      const isChecked = this.checked;

      if (isChecked) {
        settings.selected_content.world_info.selected[world] = worldEntries.map(e => e.uid);
      } else {
        delete settings.selected_content.world_info.selected[world];
      }

      // 更新所有子条目
      worldDiv.find('.wi-entry input').prop('checked', isChecked);

      Object.assign(extension_settings.vectors_enhanced, settings);
      saveSettingsDebounced();
    });

    worldDiv.append(worldHeader);

    // 条目列表
    worldEntries.forEach(entry => {
      const isChecked = selectedEntries.includes(entry.uid);

      const checkbox = $(`
                <label class="checkbox_label wi-entry flex-container alignItemsCenter">
                    <input type="checkbox" value="${entry.uid}" data-world="${world}" ${isChecked ? 'checked' : ''} />
                    <span class="flex1">${entry.comment || '(无注释)'}</span>
                </label>
            `);

      checkbox.find('input').on('change', function () {
        if (!settings.selected_content.world_info.selected[world]) {
          settings.selected_content.world_info.selected[world] = [];
        }

        if (this.checked) {
          if (!settings.selected_content.world_info.selected[world].includes(entry.uid)) {
            settings.selected_content.world_info.selected[world].push(entry.uid);
          }
        } else {
          settings.selected_content.world_info.selected[world] = settings.selected_content.world_info.selected[
            world
          ].filter(id => id !== entry.uid);
        }

        // 更新全选复选框状态
        const allChecked = worldEntries.every(e =>
          settings.selected_content.world_info.selected[world]?.includes(e.uid),
        );
        worldHeader.find('.world-select-all').prop('checked', allChecked);

        // Clean up empty world arrays
        if (settings.selected_content.world_info.selected[world].length === 0) {
          delete settings.selected_content.world_info.selected[world];
        }

        Object.assign(extension_settings.vectors_enhanced, settings);
        saveSettingsDebounced();
      });

      worldDiv.append(checkbox);
    });

    wiList.append(worldDiv);
  }
}

/**
 * Updates chat message range settings
 */
function updateChatSettings() {
  const context = getContext();
  const messageCount = context.chat?.length || 0;

  $('#vectors_enhanced_chat_start').attr('max', messageCount);
  $('#vectors_enhanced_chat_end').attr('min', -1).attr('max', messageCount);
}

// Event handlers
const onChatEvent = debounce(async () => {
  if (settings.auto_vectorize) {
    await moduleWorker.update();
  }
  // Update UI lists when chat changes
  await updateFileList();
  updateChatSettings();
  await updateTaskList();
}, debounce_timeout.relaxed);

jQuery(async () => {
  // 使用独立的设置键避免冲突
  const SETTINGS_KEY = 'vectors_enhanced';

  if (!extension_settings[SETTINGS_KEY]) {
    extension_settings[SETTINGS_KEY] = settings;
  }

  // 深度合并设置，确保所有必需的属性都存在
  Object.assign(settings, extension_settings[SETTINGS_KEY]);

  // 确保 chat types 存在（处理旧版本兼容性）
  if (!settings.selected_content.chat.types) {
    settings.selected_content.chat.types = { user: true, assistant: true };
  }

  // 确保 include_hidden 属性存在
  if (settings.selected_content.chat.include_hidden === undefined) {
    settings.selected_content.chat.include_hidden = false;
  }

  // 确保所有必需的结构都存在
  if (!settings.selected_content.chat.range) {
    settings.selected_content.chat.range = { start: 0, end: -1 };
  }

  // 确保 vector_tasks 存在
  if (!settings.vector_tasks) {
    settings.vector_tasks = {};
  }

  // 保存修正后的设置
  Object.assign(extension_settings[SETTINGS_KEY], settings);
  saveSettingsDebounced();

  // 第三方插件需要使用完整路径
  const template = await renderExtensionTemplateAsync('third-party/vectors-enhanced', 'settings');
  $('#extensions_settings2').append(template);

  // Initialize master switch first
  $('#vectors_enhanced_master_enabled')
    .prop('checked', settings.master_enabled)
    .on('change', function () {
      settings.master_enabled = $(this).prop('checked');
      Object.assign(extension_settings.vectors_enhanced, settings);
      saveSettingsDebounced();
      updateMasterSwitchState();
    });

  // Initialize master switch state
  updateMasterSwitchState();

  // Initialize UI elements
  $('#vectors_enhanced_source')
    .val(settings.source)
    .on('change', () => {
      settings.source = String($('#vectors_enhanced_source').val());
      Object.assign(extension_settings.vectors_enhanced, settings);
      saveSettingsDebounced();
      toggleSettings();
    });

  $('#vectors_enhanced_vllm_model')
    .val(settings.vllm_model)
    .on('input', () => {
      settings.vllm_model = String($('#vectors_enhanced_vllm_model').val());
      Object.assign(extension_settings.vectors_enhanced, settings);
      saveSettingsDebounced();
    });

  $('#vectors_enhanced_vllm_url')
    .val(settings.vllm_url)
    .on('input', () => {
      settings.vllm_url = String($('#vectors_enhanced_vllm_url').val());
      Object.assign(extension_settings.vectors_enhanced, settings);
      saveSettingsDebounced();
    });

  $('#vectors_enhanced_local_model')
    .val(settings.local_model)
    .on('input', () => {
      settings.local_model = String($('#vectors_enhanced_local_model').val());
      Object.assign(extension_settings.vectors_enhanced, settings);
      saveSettingsDebounced();
    });

  // Ollama settings handlers
  $('#vectors_enhanced_ollama_model')
    .val(settings.ollama_model)
    .on('input', () => {
      settings.ollama_model = String($('#vectors_enhanced_ollama_model').val());
      Object.assign(extension_settings.vectors_enhanced, settings);
      saveSettingsDebounced();
    });

  $('#vectors_enhanced_ollama_url')
    .val(settings.ollama_url)
    .on('input', () => {
      settings.ollama_url = String($('#vectors_enhanced_ollama_url').val());
      Object.assign(extension_settings.vectors_enhanced, settings);
      saveSettingsDebounced();
    });

  $('#vectors_enhanced_ollama_keep')
    .prop('checked', settings.ollama_keep)
    .on('input', () => {
      settings.ollama_keep = $('#vectors_enhanced_ollama_keep').prop('checked');
      Object.assign(extension_settings.vectors_enhanced, settings);
      saveSettingsDebounced();
    });

  $('#vectors_enhanced_auto_vectorize')
    .prop('checked', settings.auto_vectorize)
    .on('input', () => {
      settings.auto_vectorize = $('#vectors_enhanced_auto_vectorize').prop('checked');
      Object.assign(extension_settings.vectors_enhanced, settings);
      saveSettingsDebounced();
    });

  $('#vectors_enhanced_chunk_size')
    .val(settings.chunk_size)
    .on('input', () => {
      settings.chunk_size = Number($('#vectors_enhanced_chunk_size').val());
      Object.assign(extension_settings.vectors_enhanced, settings);
      saveSettingsDebounced();
    });

  $('#vectors_enhanced_overlap_percent')
    .val(settings.overlap_percent)
    .on('input', () => {
      settings.overlap_percent = Number($('#vectors_enhanced_overlap_percent').val());
      Object.assign(extension_settings.vectors_enhanced, settings);
      saveSettingsDebounced();
    });

  $('#vectors_enhanced_score_threshold')
    .val(settings.score_threshold)
    .on('input', () => {
      settings.score_threshold = Number($('#vectors_enhanced_score_threshold').val());
      Object.assign(extension_settings.vectors_enhanced, settings);
      saveSettingsDebounced();
    });

  $('#vectors_enhanced_force_chunk_delimiter')
    .val(settings.force_chunk_delimiter)
    .on('input', () => {
      settings.force_chunk_delimiter = String($('#vectors_enhanced_force_chunk_delimiter').val());
      Object.assign(extension_settings.vectors_enhanced, settings);
      saveSettingsDebounced();
    });

  $('#vectors_enhanced_enabled')
    .prop('checked', settings.enabled)
    .on('input', () => {
      settings.enabled = $('#vectors_enhanced_enabled').prop('checked');
      Object.assign(extension_settings.vectors_enhanced, settings);
      saveSettingsDebounced();
    });

  $('#vectors_enhanced_query_messages')
    .val(settings.query_messages)
    .on('input', () => {
      settings.query_messages = Number($('#vectors_enhanced_query_messages').val());
      Object.assign(extension_settings.vectors_enhanced, settings);
      saveSettingsDebounced();
    });

  $('#vectors_enhanced_max_results')
    .val(settings.max_results)
    .on('input', () => {
      settings.max_results = Number($('#vectors_enhanced_max_results').val());
      Object.assign(extension_settings.vectors_enhanced, settings);
      saveSettingsDebounced();
    });

  // 内容标签设置事件处理器
  $('#vectors_enhanced_tag_chat').on('input', () => {
    const value = $('#vectors_enhanced_tag_chat').val().trim() || 'past_chat';
    settings.content_tags.chat = value;
    Object.assign(extension_settings.vectors_enhanced, settings);
    saveSettingsDebounced();
  });

  $('#vectors_enhanced_tag_wi').on('input', () => {
    const value = $('#vectors_enhanced_tag_wi').val().trim() || 'world_part';
    settings.content_tags.world_info = value;
    Object.assign(extension_settings.vectors_enhanced, settings);
    saveSettingsDebounced();
  });

  $('#vectors_enhanced_tag_file').on('input', () => {
    const value = $('#vectors_enhanced_tag_file').val().trim() || 'databank';
    settings.content_tags.file = value;
    Object.assign(extension_settings.vectors_enhanced, settings);
    saveSettingsDebounced();
  });

  $('#vectors_enhanced_template')
    .val(settings.template)
    .on('input', () => {
      settings.template = String($('#vectors_enhanced_template').val());
      Object.assign(extension_settings.vectors_enhanced, settings);
      saveSettingsDebounced();
    });

  $('#vectors_enhanced_depth')
    .val(settings.depth)
    .on('input', () => {
      settings.depth = Number($('#vectors_enhanced_depth').val());
      Object.assign(extension_settings.vectors_enhanced, settings);
      saveSettingsDebounced();
    });

  $(`input[name="vectors_position"][value="${settings.position}"]`).prop('checked', true);
  $('input[name="vectors_position"]').on('change', () => {
    settings.position = Number($('input[name="vectors_position"]:checked').val());
    Object.assign(extension_settings.vectors_enhanced, settings);
    saveSettingsDebounced();
  });

  $('#vectors_enhanced_depth_role')
    .val(settings.depth_role)
    .on('change', () => {
      settings.depth_role = Number($('#vectors_enhanced_depth_role').val());
      Object.assign(extension_settings.vectors_enhanced, settings);
      saveSettingsDebounced();
    });

  $('#vectors_enhanced_include_wi')
    .prop('checked', settings.include_wi)
    .on('input', () => {
      settings.include_wi = $('#vectors_enhanced_include_wi').prop('checked');
      Object.assign(extension_settings.vectors_enhanced, settings);
      saveSettingsDebounced();
    });

  // Content selection handlers
  $('#vectors_enhanced_chat_enabled')
    .prop('checked', settings.selected_content.chat.enabled)
    .on('input', () => {
      settings.selected_content.chat.enabled = $('#vectors_enhanced_chat_enabled').prop('checked');
      Object.assign(extension_settings.vectors_enhanced, settings);
      saveSettingsDebounced();
      updateContentSelection();
    });

  $('#vectors_enhanced_files_enabled')
    .prop('checked', settings.selected_content.files.enabled)
    .on('input', async () => {
      settings.selected_content.files.enabled = $('#vectors_enhanced_files_enabled').prop('checked');
      Object.assign(extension_settings.vectors_enhanced, settings);
      saveSettingsDebounced();
      updateContentSelection();
      if (settings.selected_content.files.enabled) {
        await updateFileList();
      }
    });

  $('#vectors_enhanced_wi_enabled')
    .prop('checked', settings.selected_content.world_info.enabled)
    .on('input', async () => {
      settings.selected_content.world_info.enabled = $('#vectors_enhanced_wi_enabled').prop('checked');
      Object.assign(extension_settings.vectors_enhanced, settings);
      saveSettingsDebounced();
      updateContentSelection();
      if (settings.selected_content.world_info.enabled) {
        await updateWorldInfoList();
      }
    });

  // Chat settings handlers - 确保所有属性都存在
  const chatRange = settings.selected_content.chat.range || { start: 0, end: -1 };
  const chatTypes = settings.selected_content.chat.types || { user: true, assistant: true };
  const chatTags = settings.selected_content.chat.tags || '';

  $('#vectors_enhanced_chat_start')
    .val(chatRange.start)
    .on('input', () => {
      if (!settings.selected_content.chat.range) {
        settings.selected_content.chat.range = { start: 0, end: -1 };
      }
      settings.selected_content.chat.range.start = Number($('#vectors_enhanced_chat_start').val());
      Object.assign(extension_settings.vectors_enhanced, settings);
      saveSettingsDebounced();
    });

  $('#vectors_enhanced_chat_end')
    .val(chatRange.end)
    .on('input', () => {
      if (!settings.selected_content.chat.range) {
        settings.selected_content.chat.range = { start: 0, end: -1 };
      }
      settings.selected_content.chat.range.end = Number($('#vectors_enhanced_chat_end').val());
      Object.assign(extension_settings.vectors_enhanced, settings);
      saveSettingsDebounced();
    });

  // Message type checkboxes
  $('#vectors_enhanced_chat_user')
    .prop('checked', chatTypes.user)
    .on('input', () => {
      if (!settings.selected_content.chat.types) {
        settings.selected_content.chat.types = { user: true, assistant: true };
      }
      settings.selected_content.chat.types.user = $('#vectors_enhanced_chat_user').prop('checked');
      Object.assign(extension_settings.vectors_enhanced, settings);
      saveSettingsDebounced();
    });

  $('#vectors_enhanced_chat_assistant')
    .prop('checked', chatTypes.assistant)
    .on('input', () => {
      if (!settings.selected_content.chat.types) {
        settings.selected_content.chat.types = { user: true, assistant: true };
      }
      settings.selected_content.chat.types.assistant = $('#vectors_enhanced_chat_assistant').prop('checked');
      Object.assign(extension_settings.vectors_enhanced, settings);
      saveSettingsDebounced();
    });

  // Tags input
  $('#vectors_enhanced_chat_tags')
    .val(chatTags)
    .on('input', () => {
      settings.selected_content.chat.tags = String($('#vectors_enhanced_chat_tags').val());
      Object.assign(extension_settings.vectors_enhanced, settings);
      saveSettingsDebounced();
    });

  // Include hidden messages checkbox
  $('#vectors_enhanced_chat_include_hidden')
    .prop('checked', settings.selected_content.chat.include_hidden || false)
    .on('input', () => {
      if (!settings.selected_content.chat) {
        settings.selected_content.chat = {};
      }
      settings.selected_content.chat.include_hidden = $('#vectors_enhanced_chat_include_hidden').prop('checked');
      Object.assign(extension_settings.vectors_enhanced, settings);
      saveSettingsDebounced();
    });

  // Refresh buttons
  $('#vectors_enhanced_files_refresh').on('click', async () => {
    await updateFileList();
    toastr.info('文件列表已刷新');
  });

  $('#vectors_enhanced_wi_refresh').on('click', async () => {
    await updateWorldInfoList();
    toastr.info('世界信息列表已刷新');
  });

  // Initialize UI
  toggleSettings();
  updateContentSelection();
  updateChatSettings();

  // 加载内容标签设置（确保向后兼容）
  if (!settings.content_tags) {
    settings.content_tags = {
      chat: 'past_chat',
      file: 'databank',
      world_info: 'world_part',
    };
  }
  $('#vectors_enhanced_tag_chat').val(settings.content_tags.chat);
  $('#vectors_enhanced_tag_wi').val(settings.content_tags.world_info);
  $('#vectors_enhanced_tag_file').val(settings.content_tags.file);

  // Initialize lists if enabled
  if (settings.selected_content.files.enabled) {
    await updateFileList();
  }
  if (settings.selected_content.world_info.enabled) {
    await updateWorldInfoList();
  }

  // Initialize task list
  await updateTaskList();

  // Initialize hidden messages info
  updateHiddenMessagesInfo();

  // Event listeners
  eventSource.on(event_types.MESSAGE_DELETED, onChatEvent);
  eventSource.on(event_types.MESSAGE_EDITED, onChatEvent);
  eventSource.on(event_types.MESSAGE_SENT, onChatEvent);
  eventSource.on(event_types.MESSAGE_RECEIVED, onChatEvent);
  eventSource.on(event_types.MESSAGE_SWIPED, onChatEvent);
  eventSource.on(event_types.CHAT_DELETED, chatId => {
    cachedVectors.delete(chatId);
    delete settings.vector_tasks[chatId];
    Object.assign(extension_settings.vectors_enhanced, settings);
    saveSettingsDebounced();
  });
  eventSource.on(event_types.GROUP_CHAT_DELETED, chatId => {
    cachedVectors.delete(chatId);
    delete settings.vector_tasks[chatId];
    Object.assign(extension_settings.vectors_enhanced, settings);
    saveSettingsDebounced();
  });
  eventSource.on(event_types.CHAT_CHANGED, async () => {
    await updateTaskList();
    updateHiddenMessagesInfo();
  });

  // 监听聊天重新加载事件，以便在使用 /hide 和 /unhide 命令后更新
  eventSource.on(event_types.CHAT_LOADED, async () => {
    updateHiddenMessagesInfo();
  });

  // Register slash commands
  SlashCommandParser.addCommandObject(
    SlashCommand.fromProps({
      name: 'vec-preview',
      callback: async () => {
        await previewContent();
        return '';
      },
      helpString: '预览选中的向量化内容',
    }),
  );

  SlashCommandParser.addCommandObject(
    SlashCommand.fromProps({
      name: 'vec-export',
      callback: async () => {
        await exportVectors();
        return '';
      },
      helpString: '导出向量化内容到文本文件',
    }),
  );

  SlashCommandParser.addCommandObject(
    SlashCommand.fromProps({
      name: 'vec-process',
      callback: async () => {
        await vectorizeContent();
        return '';
      },
      helpString: '处理并向量化选中的内容',
    }),
  );

  registerDebugFunction('purge-vectors', '清除所有向量', '删除当前聊天的所有向量数据', async () => {
    const chatId = getCurrentChatId();
    if (!chatId) {
      toastr.error('未选择聊天');
      return;
    }
    if (await purgeVectorIndex(chatId)) {
      toastr.success('向量已清除');
    }
  });

  // 注册隐藏消息调试函数
  registerDebugFunction('debug-hidden-messages', '调试隐藏消息', '探索消息隐藏机制', debugHiddenMessages);
  registerDebugFunction('debug-slash-commands', '调试斜杠命令', '测试斜杠命令执行', debugSlashCommands);

  // 内容过滤黑名单设置
  $('#vectors_enhanced_content_blacklist').on('input', function () {
    const blacklistText = $(this).val();
    settings.content_blacklist = blacklistText
      .split('\n')
      .map(line => line.trim())
      .filter(line => line);
    Object.assign(extension_settings.vectors_enhanced, settings);
    saveSettingsDebounced();
  });

  // 内容过滤黑名单UI初始化
  $('#vectors_enhanced_content_blacklist').val(
    Array.isArray(settings.content_blacklist) ? settings.content_blacklist.join('\n') : '',
  );

  // 隐藏消息管理按钮事件处理器
  $('#vectors_enhanced_hide_range').on('click', async () => {
    const start = Number($('#vectors_enhanced_chat_start').val()) || 0;
    const end = Number($('#vectors_enhanced_chat_end').val()) || -1;
    await toggleMessageRangeVisibility(start, end, true);
    updateHiddenMessagesInfo();
  });

  $('#vectors_enhanced_unhide_range').on('click', async () => {
    const start = Number($('#vectors_enhanced_chat_start').val()) || 0;
    const end = Number($('#vectors_enhanced_chat_end').val()) || -1;
    await toggleMessageRangeVisibility(start, end, false);
    updateHiddenMessagesInfo();
  });

  $('#vectors_enhanced_show_hidden').on('click', async () => {
    await showHiddenMessages();
  });

  // 初始化隐藏消息信息显示
  updateHiddenMessagesInfo();

  // 监听聊天变化以更新隐藏消息信息
  eventSource.on(event_types.CHAT_CHANGED, () => {
    updateHiddenMessagesInfo();
  });
});

/**
 * 调试函数：探索消息隐藏机制
 */
function debugHiddenMessages() {
  const context = getContext();
  if (!context.chat || context.chat.length === 0) {
    console.log('调试：没有可用的聊天消息');
    return;
  }

  console.log('=== 开始探索消息隐藏机制 ===');
  console.log(`总消息数: ${context.chat.length}`);

  // 检查前5条消息的完整结构
  console.log('\n前5条消息的完整结构:');
  context.chat.slice(0, 5).forEach((msg, index) => {
    console.log(`\n消息 #${index}:`, msg);

    // 检查可能的隐藏属性
    const possibleHiddenProps = ['hidden', 'is_hidden', 'hide', 'isHidden', 'visible', 'is_visible'];
    console.log(`检查可能的隐藏属性:`);
    possibleHiddenProps.forEach(prop => {
      if (prop in msg) {
        console.log(`  - ${prop}: ${msg[prop]}`);
      }
    });

    // 检查 extra 对象
    if (msg.extra) {
      console.log(`  extra 对象:`, msg.extra);
    }
  });

  console.log('\n=== 探索结束 ===');
}

/**
 * 调试函数：测试斜杠命令执行
 * @returns {Promise<void>}
 */
async function debugSlashCommands() {
  console.log('=== 测试斜杠命令执行 ===');

  try {
    // 检查可能的命令执行方法
    const context = getContext();
    console.log('\n检查上下文对象:', context);

    // 方法1：直接修改消息的 is_system 属性
    if (context.chat && context.chat.length > 0) {
      console.log('\n测试直接修改消息属性:');
      const testMessage = context.chat[0];
      console.log('第一条消息的 is_system 状态:', testMessage.is_system);
      console.log('可以通过修改 is_system 属性来隐藏/显示消息');
    }

    // 方法2：查看全局函数
    const globalFunctions = Object.keys(window).filter(
      key => key.includes('hide') || key.includes('slash') || key.includes('command'),
    );
    console.log('\n相关的全局函数:', globalFunctions);

    // 方法3：检查 jQuery 事件
    console.log('\n检查消息元素的事件处理器...');
    const messageElement = $('.mes').first();
    if (messageElement.length > 0) {
      const events = $._data(messageElement[0], 'events');
      console.log('消息元素的事件:', events);
    }
  } catch (error) {
    console.error('调试斜杠命令时出错:', error);
  }

  console.log('=== 测试结束 ===');
}

/**
 * 显示当前隐藏的消息列表
 * @returns {Promise<void>}
 */
async function showHiddenMessages() {
  const hidden = getHiddenMessages();

  if (hidden.length === 0) {
    await callGenericPopup('当前没有隐藏的消息', POPUP_TYPE.TEXT, '', { okButton: '关闭' });
    return;
  }

  // 计算隐藏消息的楼层范围
  const indexes = hidden.map(msg => msg.index).sort((a, b) => a - b);
  const ranges = [];
  let start = indexes[0];
  let end = indexes[0];

  for (let i = 1; i < indexes.length; i++) {
    if (indexes[i] === end + 1) {
      end = indexes[i];
    } else {
      ranges.push(start === end ? `第${start}层` : `第${start}-${end}层`);
      start = end = indexes[i];
    }
  }
  ranges.push(start === end ? `第${start}层` : `第${start}-${end}层`);

  const rangeText = ranges.join('，');

  let html = '<div class="hidden-messages-popup">';
  html += `<h3 style="color: var(--SmartThemeQuoteColor); margin-bottom: 1rem; font-size: 1.2em; text-align: left;">已隐藏楼层：${rangeText}</h3>`;
  html +=
    '<div class="hidden-messages-all-content" style="max-height: 60vh; overflow-y: auto; padding: 1rem; background-color: var(--SmartThemeBlurTintColor); border-radius: 6px; text-align: left; white-space: pre-wrap; word-break: break-word;">';

  // 按索引排序并显示所有隐藏消息
  hidden
    .sort((a, b) => a.index - b.index)
    .forEach((msg, idx) => {
      const msgType = msg.is_user ? '用户' : 'AI';
      html += `<span style="color: var(--SmartThemeQuoteColor); font-weight: bold;">#${msg.index} - ${msgType}（${msg.name}）：</span>\n${msg.text}\n\n`;
    });

  html += '</div></div>';

  await callGenericPopup(html, POPUP_TYPE.TEXT, '', {
    okButton: '关闭',
    wide: true,
    large: true,
  });
}

/**
 * 更新隐藏消息信息显示
 */
function updateHiddenMessagesInfo() {
  const hidden = getHiddenMessages();
  const infoDiv = $('#vectors_enhanced_hidden_info');
  const countSpan = $('#vectors_enhanced_hidden_count');
  const listDiv = $('#vectors_enhanced_hidden_list');

  countSpan.text(hidden.length);

  if (hidden.length > 0) {
    infoDiv.show();
    listDiv.empty();

    // 只显示前5条隐藏消息的预览
    const preview = hidden.slice(0, 5);
    preview.forEach(msg => {
      const msgType = msg.is_user ? '用户' : 'AI';
      const item = $(`
        <div class="hidden-message-preview">
          <strong>#${msg.index}</strong> - ${msgType}: ${msg.text}
        </div>
      `);
      listDiv.append(item);
    });

    if (hidden.length > 5) {
      listDiv.append(`<div class="text-muted">...还有 ${hidden.length - 5} 条隐藏消息</div>`);
    }
  } else {
    infoDiv.hide();
  }
}

/**
 * 切换消息的隐藏状态
 * @param {number} messageIndex 消息索引
 * @param {boolean} hide 是否隐藏
 * @returns {Promise<boolean>} 是否成功
 */
async function toggleMessageVisibility(messageIndex, hide) {
  const context = getContext();
  if (!context.chat || messageIndex < 0 || messageIndex >= context.chat.length) {
    console.error('无效的消息索引:', messageIndex);
    return false;
  }

  try {
    // 修改消息的 is_system 属性
    context.chat[messageIndex].is_system = hide;

    // 触发保存
    await context.saveChat();

    // 刷新界面
    await context.reloadCurrentChat();

    return true;
  } catch (error) {
    console.error('切换消息可见性失败:', error);
    return false;
  }
}

/**
 * 批量切换消息范围的隐藏状态
 * @param {number} startIndex 开始索引
 * @param {number} endIndex 结束索引（不包含）
 * @param {boolean} hide 是否隐藏
 * @returns {Promise<void>}
 */
async function toggleMessageRangeVisibility(startIndex, endIndex, hide) {
  const context = getContext();
  if (!context.chat) {
    toastr.error('没有可用的聊天记录');
    return;
  }

  const start = Math.max(0, startIndex);
  const end = Math.min(context.chat.length, endIndex === -1 ? context.chat.length : endIndex);

  if (start >= end) {
    toastr.error('无效的消息范围');
    return;
  }

  try {
    // 批量修改消息的 is_system 属性
    for (let i = start; i < end; i++) {
      context.chat[i].is_system = hide;
    }

    // 触发保存
    await context.saveChat();

    // 刷新界面
    await context.reloadCurrentChat();

    const action = hide ? '隐藏' : '显示';
    toastr.success(`已${action}消息 #${start} 到 #${end - 1}`);
  } catch (error) {
    console.error('批量切换消息可见性失败:', error);
    toastr.error('操作失败');
  }
}

/**
 * 获取当前隐藏的消息列表
 * @returns {Array} 隐藏消息数组
 */
function getHiddenMessages() {
  const context = getContext();
  if (!context.chat) return [];

  const hidden = [];
  context.chat.forEach((msg, index) => {
    if (msg.is_system === true) {
      hidden.push({
        index: index,
        text: msg.mes ? msg.mes.substring(0, 100) + (msg.mes.length > 100 ? '...' : '') : '',
        is_user: msg.is_user,
        name: msg.name,
      });
    }
  });

  return hidden;
}
