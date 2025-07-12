# 项目架构

## 文件结构
```
vectors-enhanced/
├── index.js              # 主入口文件 (5000+行，UI逻辑逐步迁移至 src/ui/ 模块)
├── webllm.js            # WebLLM 引擎适配器
├── settings.html        # 设置界面 (500+行，待模块化)
├── style.css            # 样式文件 (待优化)
├── manifest.json        # 扩展配置文件
├── package.json         # Node.js 项目配置 (新增)
├── webpack.config.js    # Webpack 构建配置 (新增)
├── .gitignore          # Git 忽略文件 (更新)
├── src/                 # 重构后的源代码目录 (新增)
│   ├── README.md        # 源代码目录说明
│   ├── index.js        # 模块化入口文件 (新增)
│   ├── core/           # 核心业务逻辑
│   │   ├── entities/   # 实体类定义
│   │   │   ├── Content.js      # 内容实体类 (新增)
│   │   │   ├── Vector.js       # 向量实体类 (新增)
│   │   │   ├── Task.js         # 任务实体类 (新增)
│   │   │   └── README.md
│   │   ├── extractors/ # 内容提取器 (新增)
│   │   │   ├── IContentExtractor.js    # 提取器接口 (新增)
│   │   │   ├── ChatExtractor.js        # 聊天消息提取器 (新增)
│   │   │   ├── FileExtractor.js        # 文件提取器 (新增)
│   │   │   ├── WorldInfoExtractor.js   # 世界信息提取器 (新增)
│   │   │   └── README.md
│   │   ├── tasks/      # 任务系统 (新增)
│   │   │   ├── ITask.js                # 任务接口 (新增)
│   │   │   ├── BaseTask.js             # 基础任务类 (新增)
│   │   │   ├── VectorizationTask.js    # 向量化任务 (新增)
│   │   │   ├── TaskFactory.js          # 任务工厂 (新增)
│   │   │   ├── taskTypes.js            # 任务类型常量 (新增)
│   │   │   └── README.md
│   │   └── plugins/    # 插件系统 (Phase 8 - 计划中)
│   │       ├── IVectorizationPlugin.js # 插件接口定义
│   │       ├── PluginManager.js        # 插件管理器
│   │       └── PluginLoader.js         # 插件加载器
│   ├── application/    # 应用层服务 (新增)
│   │   ├── TaskManager.js              # 任务管理器 (新增)
│   │   └── TaskQueue.js                # 任务队列 (新增)
│   ├── infrastructure/ # 基础设施层
│   │   ├── ConfigManager.js            # 配置管理器 (新增)
│   │   ├── events/    # 事件系统
│   │   │   ├── EventBus.js             # 事件总线 (新增)
│   │   │   ├── eventBus.instance.js    # 事件总线实例 (新增)
│   │   │   └── README.md
│   │   ├── storage/   # 存储适配器
│   │   │   ├── StorageAdapter.js       # 向量存储API适配器 (新增)
│   │   │   └── TaskStorageAdapter.js   # 任务存储适配器 (新增)
│   │   └── api/       # API适配器
│   │       └── VectorizationAdapter.js  # 封装所有向量化源的调用 (新增)
│   ├── utils/         # 工具函数
│   │   ├── hash.js      # 哈希计算工具 (新增)
│   │   ├── textChunking.js  # 文本分块工具 (新增)
│   │   ├── contentFilter.js  # 内容过滤工具 (新增)
│   │   ├── tagExtractor.js  # 标签提取核心模块 (新增)
│   │   ├── tagScanner.js  # 标签扫描模块 (新增)
│   │   ├── tagParser.js   # 标签解析工具 (新增)
│   │   ├── chatUtils.js   # 聊天消息统一过滤工具 (新增)
│   │   └── README.md
│   ├── ui/            # UI相关模块 (新增)
│   │   ├── domUtils.js  # DOM操作工具 (新增，包含文件下载等原生DOM操作)
│   │   ├── settingsManager.js  # 设置管理器 (新增，集中管理所有设置UI初始化)
│   │   └── components/  # 可复用UI组件 (新增)
│   │       ├── ChatSettings.js      # 聊天设置UI组件 (新增)
│   │       ├── TagRulesEditor.js    # 标签规则编辑器UI组件 (新增)
│   │       ├── TaskList.js          # 任务列表UI组件 (新增)
│   │       ├── FileList.js          # 文件列表UI组件 (新增)
│   │       ├── WorldInfoList.js     # 世界信息列表UI组件 (新增)
│   │       ├── TagUI.js             # 标签相关UI逻辑管理 (新增)
│   │       ├── MessageUI.js         # 消息相关UI逻辑管理 (新增)
│   │       ├── ActionButtons.js     # 动作按钮管理 (Phase 7.1 - 已完成)
│   │       ├── SettingsPanel.js     # 设置面板架构 (Phase 7.2 - 已完成)
│   │       ├── VectorizationSettings.js  # 向量化设置 (Phase 7.2 - 已完成)
│   │       ├── QuerySettings.js     # 查询设置 (Phase 7.2 - 已完成)
│   │       ├── InjectionSettings.js # 注入设置 (Phase 7.2 - 已完成)
│   │       ├── ContentSelectionSettings.js # 内容选择设置 (Phase 7.2 - 已完成)
│   │       ├── ProgressManager.js   # 进度管理器 (Phase 7.3 - 已完成)
│   │       ├── NotificationManager.js # 通知管理器 (Phase 7.6 - 已完成)
│   │       └── README.md
│   │   ├── EventManager.js      # UI事件协调器 (Phase 7.4 - 已完成)
│   │   ├── StateManager.js      # UI状态管理器 (Phase 7.5 - 已完成)
│   │   └── styles/              # 模块化样式 (Phase 7.7 - 已完成)
│   │       ├── index.css        # 样式入口文件
│   │       ├── base.css         # 基础样式和CSS变量
│   │       ├── buttons.css      # 按钮样式
│   │       ├── content-selection.css # 内容选择样式
│   │       ├── forms.css        # 表单样式
│   │       ├── preview.css      # 预览样式
│   │       ├── progress.css     # 进度条样式
│   │       ├── tags.css         # 标签样式
│   │       ├── tasks.css        # 任务列表样式
│   │       └── responsive.css   # 响应式样式
│   ├── plugins/       # 内置插件 (Phase 8 - 计划中)
│   │   └── builtin/
│   │       ├── TransformersPlugin.js  # Transformers向量化插件
│   │       ├── OllamaPlugin.js        # Ollama向量化插件
│   │       ├── vLLMPlugin.js          # vLLM向量化插件
│   │       ├── WebLLMPlugin.js        # WebLLM向量化插件
│   │       ├── OpenAIPlugin.js        # OpenAI向量化插件
│   │       └── CoherePlugin.js        # Cohere向量化插件
│   ├── legacy/        # 将被重构的旧代码
│   │   └── README.md
│   ├── build-setup-plan.md  # 构建环境设置计划 (新增)
│   └── refactoring-progress.md  # 重构进度报告 (新增)
├── plugins/            # 外部插件目录 (Phase 8 - 计划中)
│   └── example/
│       └── MockVectorizerPlugin.js  # 示例插件
├── docs/               # 文档目录 (Phase 8 - 计划中)
│   └── plugin-development.md  # 插件开发指南
├── scripts/            # 脚本目录 (Phase 8 - 计划中)
│   └── create-plugin.js  # 插件模板生成器
├── debug/               # 调试工具目录
│   ├── debugger.js
│   ├── ui-manager.js
│   ├── analyzers/       # 分析工具
│   ├── templates/       # UI模板
│   └── tools/          # 辅助工具
└── SuperAgent/         # 项目管理目录
    ├── tech-structure.md
    ├── project-brief.md
    └── Context/
        ├── project-architecture.md
        └── iterations-log.md
```

## 依赖关系图
```
当前架构（高度耦合）：
index.js
├──> jQuery (全局依赖)
├──> SillyTavern API (getContext, eventSource等)
├──> 向量化引擎
│    ├──> Transformers.js
│    ├──> Ollama API
│    ├──> vLLM API
│    └──> WebLLM (webllm.js)
├──> UI组件 (settings.html)
│    └──> src/ui/components/TagUI.js
├──> 样式 (style.css)
└──> 调试工具 (debug/*)

目标架构（分层解耦）：
见 refactoring-architecture.md
```

## 模块描述

### 核心模块（已重构模块化）
1. **设置管理**（已重构 ✅）
   - ConfigManager：统一配置管理接口
   - SettingsManager：UI设置初始化和事件处理
   - 与extension_settings深度集成

2. **内容提取**（已重构 ✅）
   - IContentExtractor：统一提取器接口
   - ChatExtractor：聊天消息提取，集成chatUtils
   - FileExtractor：文件内容获取
   - WorldInfoExtractor：世界信息读取

3. **向量化处理**（已重构 ✅）
   - VectorizationAdapter：多引擎适配统一接口
   - 支持Transformers/Ollama/vLLM/WebLLM/OpenAI/Cohere
   - 批量处理和错误恢复机制

4. **任务管理**（已重构 ✅）
   - TaskManager：新旧任务系统协调器
   - TaskQueue：任务队列和并发控制
   - TaskFactory：任务创建和格式转换
   - 双写模式确保向后兼容

5. **向量存储**（已重构 ✅）
   - StorageAdapter：向量数据库CRUD统一接口
   - TaskStorageAdapter：任务持久化存储
   - 依赖注入避免循环引用

6. **内容注入**（部分重构）
   - 相似度匹配（保持原有逻辑）
   - 注入位置计算（保持原有逻辑）
   - 标签系统（已重构TagExtractor/TagParser/TagScanner）

7. **UI交互**（已重构 ✅）
   - 组件化UI：TaskList/FileList/WorldInfoList等
   - domUtils：统一DOM操作
   - MessageUI：消息相关UI逻辑
   - TagUI：标签相关UI管理

### UI 模块 (Phase 7重构)
- **src/ui/domUtils.js**: 封装所有DOM操作，旨在将UI逻辑与业务逻辑分离。目前已包含：
  - 基础UI状态管理：`updateContentSelection`, `updateMasterSwitchState`, `toggleSettings`
  - 进度显示：`hideProgress`, `updateProgress`
  - 文件操作：`triggerDownload` (封装了原生DOM文件下载逻辑)
- **src/ui/settingsManager.js**: 集中管理所有设置相关的UI初始化和事件处理
  - 从 index.js 的 jQuery ready 处理器中提取了约373行设置初始化代码
  - 使用 SettingsManager 类组织所有设置的初始化逻辑
  - 通过 ConfigManager 处理设置的持久化
  - Phase 9.3: 添加实验性设置管理（管道开关）
- **src/ui/components/**: 存放可复用的UI组件
- **src/ui/ProgressManager.js**: 集中进度条管理（Phase 7.3）
- **src/ui/EventManager.js**: UI事件协调器（Phase 7.4）
- **src/ui/StateManager.js**: UI状态管理器（Phase 7.5）

### 文本处理管道模块 (Phase 9已完成)
- **src/core/pipeline/ITextProcessor.js**: 文本处理器抽象接口，支持生命周期管理
- **src/core/pipeline/TextPipeline.js**: 核心管道，集成中间件、生命周期和事件系统
- **src/core/pipeline/ProcessorRegistry.js**: 处理器注册表
- **src/core/pipeline/TextDispatcher.js**: 路由文本到合适的处理器
- **src/core/pipeline/ProcessingContext.js**: 携带上下文信息
- **src/core/pipeline/processors/VectorizationProcessor.js**: 向量化处理器包装器（Phase 9.2）
- **src/core/pipeline/adapters/ExtractorPipeline.js**: 内容提取器管道适配器（Phase 9.2）
- **src/core/pipeline/ProcessorFactory.js**: 处理器工厂（Phase 9.2）
- **src/core/pipeline/PipelineIntegration.js**: 管道集成助手（Phase 9.2）

#### 中间件系统 (Phase 9.4.1)
- **src/core/pipeline/middleware/IMiddleware.js**: 中间件接口定义
- **src/core/pipeline/middleware/LoggingMiddleware.js**: 日志记录中间件
- **src/core/pipeline/middleware/ValidationMiddleware.js**: 输入验证中间件
- **src/core/pipeline/middleware/TransformMiddleware.js**: 数据转换中间件
- **src/core/pipeline/MiddlewareManager.js**: 中间件管理器

#### 生命周期管理 (Phase 9.4.2)
- **src/core/pipeline/LifecycleManager.js**: 处理器和中间件生命周期管理
- **扩展的ITextProcessor**: 支持start/stop/pause/resume/healthCheck等生命周期方法

#### 事件系统 (Phase 9.4.3)
- **src/core/pipeline/events/PipelineEventBus.js**: 管道专用事件总线
- **src/core/pipeline/events/EventListenerFactory.js**: 事件监听器工厂
- **集成事件支持**: TextPipeline和LifecycleManager完全集成事件系统

### 并行实现策略 (Phase 9.3)
- **performVectorizationPipeline**: 新的管道版本向量化函数
- **settings.use_pipeline**: 功能开关标志
- **实验性功能UI**: 在settings-modular.html中添加了管道开关
- **A/B测试支持**: window.vectorsPipelineABTest()全局函数
- **安全降级**: 出错时自动回退到原实现
  - **ActionButtons.js**: 主要操作按钮组件 (Phase 7.1新增)
    - 集中管理Preview/Export/Vectorize/Abort按钮处理逻辑
    - 统一按钮状态管理和错误处理
    - 智能按钮切换(向量化↔中断)
  - **SettingsPanel.js**: 设置面板架构组件 (Phase 7.2新增)
    - 统一模板加载和子组件协调
    - 提供子组件管理接口
  - **VectorizationSettings.js**: 向量化设置组件 (Phase 7.2新增)
    - 向量化源选择(Transformers/vLLM/Ollama)
    - 模型配置和参数设置
    - 源配置验证和联动逻辑
  - **QuerySettings.js**: 查询设置组件 (Phase 7.2新增)
    - Rerank配置管理
    - API参数验证
    - 设置状态同步
  - **InjectionSettings.js**: 注入设置组件 (Phase 7.2新增)
    - 注入模板和位置配置
    - 内容标签管理
    - 深度设置控制
  - **ProgressManager.js**: 进度条管理器 (Phase 7.3新增)
    - 集中管理所有进度条显示逻辑
    - 处理向量化过程中的阶段性进度更新
    - 自动隐藏进度条的bug修复
  - **EventManager.js**: 事件管理器 (Phase 7.4新增)
    - 集中管理所有事件绑定逻辑
    - 使用委托事件处理动态内容
    - 修复了事件绑定丢失的问题
  - **StateManager.js**: UI状态管理器 (Phase 7.5新增)
    - 集中管理UI元素的启用/禁用状态
    - 处理向量化过程中的按钮状态切换
    - 统一的状态同步机制
  - **MessageUI.js**: 管理与消息显示相关的UI逻辑，如隐藏消息提示、内容预览弹窗等
  - **ChatSettings.js**: 聊天设置UI组件
  - **TagRulesEditor.js**: 标签规则编辑器UI组件
  - **TaskList.js**: 任务列表UI组件
  - **FileList.js**: 文件列表UI组件
  - **WorldInfoList.js**: 世界信息列表UI组件
  - **TagUI.js**: 标签相关UI逻辑管理

### 辅助模块
- **webllm.js**: WebLLM引擎的简单封装
- **debug/**: 开发调试工具集
- **src/utils/tagScanner.js**: 包含 `scanTextForTags` 函数，负责在UI中扫描和识别文本中的标签，以便进行高亮或其他界面操作。
- **src/utils/tagParser.js**: 提供解析标签配置的工具函数，特别是处理带有排除规则的复杂标签字符串（例如 "include,tags - exclude,tags"）。
- **src/utils/chatUtils.js**: 统一的消息过滤工具，提供 `getMessages` 等函数，消除了之前在 UI 和数据处理层的重复逻辑。
- **src/infrastructure/storage/StorageAdapter.js**: 存储适配器，封装所有向量存储相关的 API 调用，使用依赖注入模式避免循环引用。
- **src/infrastructure/api/VectorizationAdapter.js**: 向量化 API 适配器，统一封装所有向量化源（Transformers、Ollama、vLLM、WebLLM、OpenAI、Cohere）的调用接口。

### 任务系统模块 (Phase 6 完成)
- **src/core/tasks/**: 任务系统核心
  - **ITask.js**: 任务接口定义，规范所有任务的基本方法
  - **BaseTask.js**: 基础任务类，提供通用任务功能和状态管理
  - **VectorizationTask.js**: 向量化任务实现，支持新旧格式转换
  - **TaskFactory.js**: 任务工厂，负责任务创建、类型识别和格式转换
- **src/application/**: 应用层服务
  - **TaskManager.js**: 任务管理器，协调新旧任务系统，实现双写模式
  - **TaskQueue.js**: 任务队列，支持优先级、并发控制和取消机制
- **src/infrastructure/storage/TaskStorageAdapter.js**: 任务存储适配器，处理新旧格式任务的持久化

### 内容提取系统 (Phase 5 完成)
- **src/core/extractors/**: 内容提取器系统
  - **IContentExtractor.js**: 提取器统一接口
  - **ChatExtractor.js**: 聊天消息提取器，集成统一的消息过滤逻辑
  - **FileExtractor.js**: 文件内容提取器，处理各种文件格式
  - **WorldInfoExtractor.js**: 世界信息提取器，按世界和条目组织内容

## 重构进展状态

### 已解决的架构问题 ✅
1. **单一职责违反**: 已通过模块化重构大幅改善
   - UI逻辑迁移至 `src/ui/components/`
   - 业务逻辑迁移至 `src/core/`
   - 基础设施迁移至 `src/infrastructure/`

2. **高耦合度**: 已大幅改善
   - UI与业务逻辑分离（domUtils、各UI组件）
   - 依赖注入模式避免循环引用
   - 适配器模式统一外部接口

3. **扩展困难**: 已解决
   - 插件式架构（提取器、任务、适配器）
   - 工厂模式支持新类型扩展
   - 统一接口便于功能添加

4. **测试困难**: 已大幅改善
   - 模块化代码便于单元测试
   - 依赖注入支持Mock测试
   - 接口规范化

5. **代码重复**: 已解决
   - 统一工具函数（chatUtils、tagExtractor等）
   - 复用组件（UI components）
   - 统一适配器模式

6. **消息过滤逻辑不一致**: 已解决 ✅
   - 创建统一的 `chatUtils.js` 模块
   - 所有消息过滤使用统一接口
   - 消除了UI和数据层的重复逻辑

### Bug修复记录
1. **EventManager事件绑定问题**: 已修复 ✅
   - 问题：动态生成的任务列表项按钮事件绑定失效
   - 原因：使用直接事件绑定而非委托事件
   - 解决：改用jQuery委托事件处理动态内容

2. **进度条自动隐藏问题**: 已修复 ✅
   - 问题：向量化完成后进度条未自动隐藏
   - 原因：ProgressManager未正确调用hideProgress
   - 解决：在处理结束时确保调用隐藏方法

3. **任务名称逻辑问题**: 已修复 ✅
   - 问题：任务名称显示不正确
   - 原因：使用了错误的属性名
   - 解决：统一使用task.name属性

### 性能优化记录
1. **代码去重分析**: 已完成 ✅
   - 创建了deduplication-analysis.md文档
   - 识别了6个主要重复区域
   - 计划通过模块化减少约30-40%的代码量

### 待完成的架构改进
1. **完全模块化入口**: 创建新的模块化入口文件
2. **插件系统**: 实现完整的插件架构
3. **样式模块化**: CSS拆分和组织
4. **性能优化**: 缓存和批处理优化

## 当前系统状态

### TaskManager系统运行状态
- **运行模式**: TaskManager（新任务系统）
- **向后兼容**: 完全保持，所有现有任务正常访问
- **系统健康**: 全部模块正常运行
- **状态检查**: 使用 `vectorsTaskSystemStatus()` 可查看详细状态

### 重构完成度
- **Phase 0-1**: 基础设施层 ✅ 100%
- **Phase 2**: 核心实体 ✅ 100% 
- **Phase 3**: 工具函数提取 ✅ 100%
- **Phase 4**: 适配器层 ✅ 100%
- **Phase 5**: 内容提取器 ✅ 100%
- **Phase 6**: 任务系统 ✅ 100%
- **Phase 7**: UI层重构 ✅ 95% (ActionButtons + SettingsPanel架构 + ProgressManager/EventManager/StateManager完成)
- **Phase 8**: 任务系统兼容性扩展 📋 待实施
- **Phase 9**: 文本处理管道 ✅ 100%
  - **Phase 9.1**: 管道架构基础 ✅ 100%
  - **Phase 9.2**: 适配器模式集成 ✅ 100%
  - **Phase 9.3**: 并行实现 ✅ 100%
  - **Phase 9.4**: 扩展性支持 ✅ 100%
    - **Phase 9.4.1**: 中间件系统 ✅ 100%
    - **Phase 9.4.2**: 生命周期管理 ✅ 100%
    - **Phase 9.4.3**: 事件支持 ✅ 100%
- **Phase 9.5**: 安全验证和迁移 📋 待实施
- **Phase 10**: 架构切换 📋 待实施  
- **Phase 11**: 清理优化和新功能 📋 待实施

### 关键特性
1. **双系统兼容**: 新任务系统与legacy系统完美共存
2. **零停机迁移**: 重构过程中功能始终可用
3. **渐进式架构**: 每个阶段都是可独立验证的改进
4. **向后兼容**: 所有现有数据和功能完全保留
