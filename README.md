# PPO Trading

基于PPO（Proximal Policy Optimization）强化学习算法的A股交易策略回测系统。

## 项目简介

本项目使用Stable Baselines3库实现PPO算法，结合技术指标（MACD等）和市场数据，构建了一个完整的股票交易策略回测框架。系统支持动态选股、滚动训练和回测评估。

## 主要功能

- **强化学习交易**：使用PPO算法学习最优交易策略
- **动态选股**：基于MACD金叉信号和相关性分析进行股票选择
- **滚动训练**：支持历史数据滚动窗口训练和测试
- **回测评估**：提供夏普比率、最大回撤、卡玛比率等关键指标
- **技术指标**：集成MACD、移动平均等技术分析工具

## 项目结构

```
ppo_trading/
├── run_backtest.py          # 回测主程序
├── rolling_train.py         # 滚动训练函数
├── trading_env.py            # 交易环境（Gymnasium）
├── dynamic_scorer.py         # 动态选股器
├── csv_loader.py             # CSV数据加载器
├── indicators.py             # 技术指标计算
├── math_utils.py             # 数学工具函数
├── requirements.txt          # 依赖包列表
├── .gitignore               # Git忽略文件
└── README.md                # 项目说明文档
```

## 安装说明

### 环境要求

- Python 3.9+
- CUDA（可选，用于GPU加速）

### 安装步骤

1. 克隆项目
```bash
git clone <repository-url>
cd ppo_trading
```

2. 创建虚拟环境（推荐）
```bash
conda create -n ppo_trading python=3.9
conda activate ppo_trading
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

## 使用方法

### 数据准备

将A股历史数据CSV文件放入`data/`目录，CSV文件需包含以下列：
- `trade_date`: 交易日期
- `close`: 收盘价
- 其他必要的技术指标列

### 运行回测

```bash
python run_backtest.py
```

### 配置参数

可以在`run_backtest.py`中调整以下参数：
- `data_dir`: 数据目录路径
- `start_date`: 回测开始日期
- `end_date`: 回测结束日期
- `total_timesteps`: PPO训练步数

## 核心组件

### 交易环境 (AShareTradingEnv)

基于Gymnasium框架的自定义交易环境，提供：
- 状态空间：包含股票技术指标和市场特征
- 动作空间：31维连续动作（30只股票权重 + 现金权重）
- 奖励函数：综合考虑收益率、回撤和交易成本

### 动态选股器 (dynamic_select)

基于以下策略进行股票选择：
- MACD金叉信号识别
- 历史金叉后的平均收益率
- 相关性过滤（避免选择高相关股票）
- 动态权重调整

### 滚动训练 (rolling_train)

实现滚动窗口训练策略：
- 历史数据训练PPO模型
- 测试期评估模型性能
- 收集equity曲线用于回测分析

## 回测指标

系统计算以下关键指标：
- **夏普比率 (Sharpe Ratio)**: 风险调整后收益
- **最大回撤 (Max Drawdown)**: 最大亏损幅度
- **卡玛比率 (Calmar Ratio)**: 年化收益/最大回撤

## 依赖包

主要依赖包括：
- `numpy`: 数值计算
- `pandas`: 数据处理
- `stable-baselines3`: 强化学习算法
- `torch`: 深度学习框架
- `gymnasium`: 强化学习环境
- 其他辅助库详见`requirements.txt`

## 注意事项

1. **设备选择**: 对于MlpPolicy，建议使用CPU而非GPU，以提高训练效率
2. **数据质量**: 确保CSV数据完整且格式正确
3. **训练时间**: 完整回测可能需要较长时间，建议先用小数据集测试
4. **参数调优**: 根据实际数据调整训练参数和模型超参数

## 常见问题

### Q: 如何修改训练参数？
A: 在`rolling_train.py`中修改`model.learn(total_timesteps=...)`参数

### Q: 如何添加新的技术指标？
A: 在`indicators.py`中添加指标计算函数，并在`csv_loader.py`中调用

### Q: 如何处理缺失数据？
A: 系统已内置NaN处理机制，会自动填充缺失值

## 许可证

本项目仅供学习和研究使用。

## 联系方式

如有问题或建议，请通过以下方式联系：
- Email: 1226240686@qq.com

## 更新日志

### v1.0.0
- 初始版本发布
- 实现PPO交易策略
- 支持动态选股和回测评估
