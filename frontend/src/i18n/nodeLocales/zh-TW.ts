import type { NodeTranslations } from './types';

const zhTW: NodeTranslations = {
  // ── CNN ──
  Conv2d: {
    description: '對輸入張量套用 2D 卷積（封裝 nn.Conv2d）',
    params: {
      in_channels: '輸入通道數',
      out_channels: '輸出通道數',
      kernel_size: '卷積核大小',
      stride: '卷積步幅',
      padding: '兩側的零填充',
    },
  },
  MaxPool2d: {
    description: '對輸入張量套用 2D 最大池化（封裝 nn.MaxPool2d）',
    params: {
      kernel_size: '池化視窗大小',
      stride: '池化視窗步幅',
    },
  },
  BatchNorm2d: {
    description: '對輸入張量套用 2D 批次正規化（封裝 nn.BatchNorm2d）',
    params: {
      num_features: '要正規化的特徵（通道）數量',
    },
  },
  Dropout: {
    description: '對輸入張量套用 Dropout 正則化（封裝 nn.Dropout）',
    params: {
      p: '元素被歸零的機率',
    },
  },
  Activation: {
    description: '對輸入張量套用激活函數（ReLU、Sigmoid 或 Tanh）',
    params: {
      function: '要套用的激活函數',
    },
  },

  // ── RNN ──
  LSTM: {
    description: '對輸入序列套用 LSTM 遞迴層（封裝 nn.LSTM）',
    params: {
      input_size: '輸入的預期特徵數',
      hidden_size: '隱藏狀態的特徵數',
      num_layers: '遞迴層數量',
      batch_first: '若為 True，輸入/輸出形狀為 (batch, seq, feature)',
      bidirectional: '若為 True，則為雙向 LSTM',
    },
  },
  GRU: {
    description: '對輸入序列套用 GRU 遞迴層（封裝 nn.GRU）',
    params: {
      input_size: '輸入的預期特徵數',
      hidden_size: '隱藏狀態的特徵數',
      num_layers: '遞迴層數量',
      batch_first: '若為 True，輸入/輸出形狀為 (batch, seq, feature)',
      bidirectional: '若為 True，則為雙向 GRU',
    },
  },

  // ── Transformer ──
  MultiHeadAttention: {
    description: '套用多頭注意力機制（封裝 nn.MultiheadAttention）',
    params: {
      embed_dim: '模型的總維度',
      num_heads: '平行注意力頭的數量',
    },
  },
  TransformerEncoder: {
    description: '對輸入張量套用 Transformer 編碼器堆疊',
    params: {
      d_model: '模型維度',
      nhead: '注意力頭的數量',
      num_layers: '編碼器層數',
      dim_feedforward: '前饋網路維度',
    },
  },
  TransformerDecoder: {
    description: '對輸入張量套用 Transformer 解碼器堆疊（含編碼器記憶）',
    params: {
      d_model: '模型維度',
      nhead: '注意力頭的數量',
      num_layers: '解碼器層數',
      dim_feedforward: '前饋網路維度',
    },
  },

  // ── RL ──
  DQN: {
    description: '建立用於強化學習的深度 Q 網路（簡單 MLP）',
    params: {
      state_dim: '狀態空間維度',
      action_dim: '動作空間維度',
      hidden_dim: '隱藏層維度',
    },
  },
  PPO: {
    description: '建立用於強化學習的 PPO Actor-Critic 網路',
    params: {
      state_dim: '狀態空間維度',
      action_dim: '動作空間維度',
      hidden_dim: '隱藏層維度',
    },
  },
  EnvWrapper: {
    description: '建立並封裝 Gymnasium 環境，回傳環境與初始觀測值',
    params: {
      env_name: 'Gymnasium 環境 ID',
    },
  },

  // ── Data ──
  Dataset: {
    description: '載入標準資料集（MNIST、CIFAR10 或 FashionMNIST）',
    params: {
      name: '要載入的資料集',
      split: '資料分割',
      data_dir: '下載/儲存資料集的目錄',
    },
  },
  DataLoader: {
    description: '將資料集包裝為 DataLoader 以進行批次迭代',
    params: {
      batch_size: '每批次的樣本數',
      shuffle: '每個 epoch 是否隨機打亂資料',
      num_workers: '資料載入的子程序數量',
    },
  },
  Transform: {
    description: '對資料集套用常見影像變換（調整大小、正規化、轉為張量）',
    params: {
      resize: '調整大小維度（0 表示不調整）',
      normalize: '套用正規化（mean=0.5, std=0.5）',
      to_tensor: '將 PIL 影像轉為張量',
    },
  },

  // ── Training ──
  Optimizer: {
    description: '建立優化器（Adam、SGD 或 AdamW）用於模型參數',
    params: {
      type: '優化器演算法',
      lr: '學習率',
      weight_decay: '權重衰減（L2 懲罰）',
    },
  },
  Loss: {
    description: '建立損失函數（CrossEntropyLoss、MSELoss 或 BCEWithLogitsLoss）',
    params: {
      type: '損失函數類型',
    },
  },
  TrainingLoop: {
    description: '在 DataLoader 上執行訓練迴圈，指定訓練 epoch 數量',
    params: {
      epochs: '訓練 epoch 數量',
      device: '訓練裝置',
    },
  },

  // ── IO ──
  ImageReader: {
    description: '從磁碟讀取影像檔案，輸出為張量 (C, H, W)，值域 [0, 1]',
    params: {
      path: '影像檔案路徑（PNG、JPEG、BMP 等）',
      mode: '載入影像的色彩模式',
      resize: '將短邊調整為此值（0 = 不調整）',
    },
  },
  ImageWriter: {
    description: '將張量儲存為影像檔案（PNG、JPEG 等）',
    params: {
      path: '輸出檔案路徑',
      format: '影像格式',
    },
  },
  ImageBatchReader: {
    description: '從目錄讀取所有影像，堆疊為批次張量 (N, C, H, W)',
    params: {
      directory: '包含影像檔案的目錄',
      pattern: '檔案比對模式（如 *.png、*.jpg）',
      resize: '將所有影像調整為此正方形大小（批次處理必需）',
      max_images: '最大載入影像數（0 = 全部）',
      mode: '色彩模式',
    },
  },
  FileReader: {
    description: '讀取文字或 CSV 檔案，輸出內容為字串或張量（數值 CSV）',
    params: {
      path: '檔案路徑',
      mode: '讀取方式',
      encoding: '文字編碼',
      csv_header: 'CSV 是否有標頭列（載入為張量時跳過）',
    },
  },

  // ── Control ──
  Compare: {
    description: '比較兩個純量值，輸出 1.0（真）或 0.0（假）',
    params: {
      operation: '比較運算子',
    },
  },
  If: {
    description: '根據條件選擇兩個輸入之一。非零 = 真，零 = 假。注意：兩個分支都會被執行。',
  },
  ForLoop: {
    description: '執行子圖（預設模組）N 次。每次迭代將上一次的輸出作為下一次的輸入。',
    params: {
      subgraph: '每次迭代要執行的子圖/預設模組名稱',
      iterations: '迭代次數',
    },
  },

  // ── Utility ──
  Print: {
    description: '將輸入值印出到主控台並傳遞',
    params: {
      label: '標籤前綴',
    },
  },
  Reshape: {
    description: '將張量重塑為指定形狀',
    params: {
      shape: '目標形狀，以逗號分隔的整數（例如 \'-1,784\'）',
    },
  },
  Concat: {
    description: '沿指定維度串接兩個張量',
    params: {
      dim: '串接的維度',
    },
  },
  Visualize: {
    description: '將資料（張量、損失值等）生成 matplotlib 圖表，輸出為 base64 編碼的 PNG',
    params: {
      title: '圖表標題',
      plot_type: '要生成的圖表類型',
    },
  },

  // ── Custom ──
  AddScalar: {
    description: '將純量值加到張量上（自訂節點範例）',
    params: {
      value: '要加的純量值',
    },
  },
};

export default zhTW;
