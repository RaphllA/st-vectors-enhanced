# Vectors Enhanced 插件架构文档

## 一、当前架构状态（Phase 8 后）

### 1.1 核心层 (core/)

#### 实体层 (entities/)
- ✅ **Content** - 内容实体，表示提取的文本
- ✅ **Vector** - 向量实体，表示向量化结果
- ✅ **Task** - 任务实体，支持版本管理

#### 提取器层 (extractors/)
- ✅ **IContentExtractor** - 统一提取器接口
- ✅ **ChatExtractor** - 聊天消息提取
- ✅ **FileExtractor** - 文件内容提取
- ✅ **WorldInfoExtractor** - 世界信息提取

#### 任务系统 (tasks/)
- ✅ **BaseTask** - 任务基类（v1.0 版本支持）
- ✅ **VectorizationTask** - 向量化任务实现
- ✅ **TaskFactory** - 任务工厂（支持类型注册）
- ⚠️ **缺失** - 其他任务类型（Rerank、Summary等）

### 1.2 基础设施层 (infrastructure/)

#### 事件系统 (events/)
- ✅ **EventBus** - 事件总线实现
- ✅ **全局实例** - eventBus.instance.js

#### 存储层 (storage/)
- ✅ **StorageAdapter** - 通用存储接口
- ✅ **TaskStorageAdapter** - 任务存储（版本兼容）

#### API层 (api/)
- ✅ **VectorizationAdapter** - 向量化API（6种源）
- ⚠️ **缺失** - 通用API框架

### 1.3 应用层 (application/)
- ✅ **TaskManager** - 任务管理器
- ✅ **TaskQueue** - 任务队列

### 1.4 UI层 (ui/)
- ✅ **组件化** - 各功能UI组件
- ✅ **设置管理** - settingsManager
- ⚠️ **耦合** - 与主文件index.js耦合较深

## 二、待实现的文本处理管道架构

### 2.1 管道系统设计

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│   Sources   │ --> │  Extractors  │ --> │  Pipeline   │ --> │  Processors  │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
      │                    │                     │                     │
   Chat UI            Extract Text          Route & Filter       Vectorization
   Files              Normalize             Transform           Rerank
   World Info         Validate              Dispatch            Summary
```

### 2.2 核心组件设计

#### TextPipeline（文本处理管道）
```javascript
class TextPipeline {
    constructor() {
        this.processors = new Map();
        this.middlewares = [];
    }
    
    // 注册处理器
    registerProcessor(type, processor) {}
    
    // 添加中间件
    use(middleware) {}
    
    // 处理文本
    async process(input) {
        // 1. 提取内容
        // 2. 应用中间件
        // 3. 路由到处理器
        // 4. 返回结果
    }
}
```

#### ProcessorRegistry（处理器注册表）
```javascript
class ProcessorRegistry {
    // 向量化处理器
    registerVectorization(adapter) {}
    
    // Rerank处理器
    registerRerank(adapter) {}
    
    // 总结处理器
    registerSummary(adapter) {}
    
    // 获取处理器
    getProcessor(type, config) {}
}
```

#### TextDispatcher（文本分发器）
```javascript
class TextDispatcher {
    // 根据任务类型分发
    async dispatch(content, taskType, config) {}
    
    // 批量分发
    async batchDispatch(contents, tasks) {}
    
    // 链式处理
    async chain(content, taskChain) {}
}
```

### 2.3 集成方案

#### 第一阶段：创建管道基础设施
1. 实现 ITextProcessor 接口
2. 创建 TextPipeline 类
3. 创建 ProcessorRegistry
4. 创建 TextDispatcher

#### 第二阶段：迁移现有功能
1. 将 VectorizationAdapter 包装为 VectorizationProcessor
2. 创建 ExtractorAdapter 连接现有提取器
3. 重构 performVectorization 使用管道

#### 第三阶段：扩展新功能
1. 实现 RerankProcessor
2. 实现 SummaryProcessor
3. 实现任务链功能

## 三、实现计划

### Phase 9: 文本处理管道实现

#### 9.1 管道基础设施
- [ ] 创建 pipeline/ 目录结构
- [ ] 实现 ITextProcessor 接口
- [ ] 实现 TextPipeline 核心类
- [ ] 实现 ProcessorRegistry
- [ ] 实现 TextDispatcher

#### 9.2 适配器层
- [ ] 创建 VectorizationProcessor 适配器
- [ ] 创建 ExtractorPipeline 连接提取器
- [ ] 创建 ProcessingContext 上下文类

#### 9.3 迁移现有功能
- [ ] 重构 performVectorization 使用管道
- [ ] 迁移任务创建逻辑
- [ ] 更新 UI 调用

#### 9.4 测试与验证
- [ ] 确保向量化功能正常
- [ ] 验证任务兼容性
- [ ] 性能测试

### Phase 10: 新功能实现（未来）

#### 10.1 Rerank功能
- [ ] 实现 RerankTask
- [ ] 实现 RerankProcessor
- [ ] 添加 Rerank API 适配器

#### 10.2 总结功能
- [ ] 实现 SummaryTask
- [ ] 实现 SummaryProcessor
- [ ] 实现提示词模板系统

#### 10.3 自动化功能
- [ ] 实现触发器系统
- [ ] 实现批处理调度
- [ ] 实现消息状态管理

#### 10.4 任务导出/导入
- [ ] 实现任务序列化
- [ ] 实现跨聊天迁移
- [ ] 实现权限管理

## 四、技术债务和改进点

### 需要解决的问题
1. **index.js 过度复杂** - 需要进一步模块化
2. **UI与逻辑耦合** - 需要完全分离
3. **缺少单元测试** - 需要添加测试框架
4. **缺少类型定义** - 考虑添加 JSDoc 或 TypeScript

### 性能优化
1. **批量处理优化** - 减少API调用
2. **缓存机制** - 避免重复处理
3. **并发控制** - 优化大量任务处理

### 可维护性改进
1. **错误处理统一** - 创建错误处理中间件
2. **日志系统增强** - 添加日志级别和持久化
3. **配置管理改进** - 支持配置验证和迁移

## 五、版本规划

### v2.0（当前开发）
- ✅ 模块化重构
- ✅ 任务系统改造
- 🚧 文本处理管道
- ⏳ 基础新功能

### v2.1（计划中）
- ⏳ Rerank功能
- ⏳ 总结功能
- ⏳ 自动化任务

### v2.2（未来）
- ⏳ 任务导出/导入
- ⏳ 高级配置
- ⏳ 插件扩展系统