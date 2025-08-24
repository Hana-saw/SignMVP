# 🤟 SignMVP Web App

リアルタイム手話認識とデータ収集のWebアプリケーション

## 🌟 機能

### 認識モード
- リアルタイム手話認識
- MediaPipeを使用した手・顔・体のランドマーク検出
- 認識結果の音声出力

### データ収集モード
- 新しい手話の登録
- 複数サンプルの一括収集
- GitHubへの自動データ保存

## 🚀 セットアップ

### 1. GitHub Pagesの有効化

1. GitHubリポジトリの設定ページに移動
2. "Pages" セクションで "Source" を "Deploy from a branch" に設定
3. "Branch" を "main" に設定
4. "Folder" を "/ (root)" に設定
5. "Save" をクリック

### 2. GitHub Tokenの設定

1. GitHub Settings > Developer settings > Personal access tokens
2. "Generate new token" をクリック
3. 以下の権限を付与：
   - `repo` (Full control of private repositories)
   - `workflow` (Update GitHub Action workflows)
4. トークンをコピーして安全な場所に保存

### 3. 環境変数の設定

GitHubリポジトリの設定で以下のシークレットを追加：

```
GITHUB_TOKEN=your_personal_access_token
```

### 4. ファイルの配置

```
SignMVP/
├── web_app/
│   ├── index.html
│   ├── styles.css
│   ├── app.js
│   └── README.md
├── web_data/          # 収集されたデータの保存先
├── models/            # 学習済みモデル
└── .github/
    └── workflows/
        └── data-processing.yml
```

## 📁 ファイル構成

### フロントエンド
- `index.html` - メインHTMLファイル
- `styles.css` - スタイルシート
- `app.js` - JavaScriptアプリケーション

### バックエンド（GitHub Actions）
- `.github/workflows/data-processing.yml` - データ処理ワークフロー

## 🔧 カスタマイズ

### モデルの統合

1. TensorFlow.jsモデルを `models/` フォルダに配置
2. `app.js` の `predictSign()` 関数を更新：

```javascript
async function predictSign(features) {
    // TensorFlow.jsモデルの読み込み
    const model = await tf.loadLayersModel('models/sign_model.json');
    
    // 特徴量の前処理
    const input = tf.tensor2d([features]);
    
    // 予測
    const prediction = await model.predict(input);
    const probabilities = await prediction.data();
    
    // 結果の後処理
    const maxIndex = probabilities.indexOf(Math.max(...probabilities));
    const labels = ['こんにちは', 'ありがとう', 'さようなら', 'other'];
    
    return {
        label: labels[maxIndex],
        confidence: probabilities[maxIndex]
    };
}
```

### データ形式の変更

収集されるデータの形式を変更する場合、`app.js` の `captureSample()` 関数を更新：

```javascript
const sampleData = {
    timestamp: Date.now(),
    signWord: currentSignWord,
    features: features,
    metadata: {
        userAgent: navigator.userAgent,
        screenResolution: `${screen.width}x${screen.height}`,
        timezone: Intl.DateTimeFormat().resolvedOptions().timeZone
    }
};
```

## 📊 データ収集の仕組み

### 1. フロントエンド
- MediaPipeを使用してランドマークを抽出
- 特徴量ベクトルを作成
- JSON形式でデータを構造化

### 2. GitHub API
- 収集されたデータをGitHubリポジトリに保存
- コミットメッセージでデータの説明を記録

### 3. GitHub Actions
- 新しいデータが追加されると自動実行
- データの前処理と検証
- モデルの再学習（オプション）

## 🔒 セキュリティ

### プライバシー保護
- 画像データは保存されません
- ランドマーク座標のみを保存
- 個人情報は含まれません

### データ検証
- 特徴量の次元チェック
- 異常値の検出
- 重複データの除去

## 📈 パフォーマンス最適化

### フロントエンド
- Web Workers を使用した並列処理
- メモリリークの防止
- フレームレートの最適化

### バックエンド
- バッチ処理による効率化
- キャッシュの活用
- 非同期処理の最適化

## 🐛 トラブルシューティング

### よくある問題

1. **カメラが起動しない**
   - HTTPS接続を確認
   - ブラウザの権限設定を確認
   - 他のアプリケーションがカメラを使用していないか確認

2. **MediaPipeが読み込まれない**
   - インターネット接続を確認
   - CDNのURLが正しいか確認
   - ブラウザのコンソールでエラーを確認

3. **データ保存に失敗**
   - GitHub Tokenの権限を確認
   - リポジトリの設定を確認
   - ネットワーク接続を確認

## 🤝 貢献

1. このリポジトリをフォーク
2. 機能ブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 🙏 謝辞

- [MediaPipe](https://mediapipe.dev/) - ランドマーク検出ライブラリ
- [TensorFlow.js](https://www.tensorflow.org/js) - 機械学習ライブラリ
- [GitHub Pages](https://pages.github.com/) - ホスティングサービス

## 📞 サポート

問題や質問がある場合は、GitHubのIssuesページでお知らせください。
