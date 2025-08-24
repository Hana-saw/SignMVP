// グローバル変数
let camera = null;
let hands = null;
let faceMesh = null;
let pose = null;
let isRecognitionActive = false;
let isCollectionActive = false;
let collectedData = [];
let currentSignWord = '';
let targetSamples = 0;
let currentSample = 0;

// タブ切り替え
function switchTab(tabName) {
    // すべてのタブを非アクティブ
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // 選択されたタブをアクティブ
    document.getElementById(tabName).classList.add('active');
    event.target.classList.add('active');
}

// MediaPipe初期化
async function initializeMediaPipe() {
    // Hands
    hands = new Hands({
        locateFile: (file) => {
            return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
        }
    });
    
    hands.setOptions({
        maxNumHands: 2,
        modelComplexity: 1,
        minDetectionConfidence: 0.7,
        minTrackingConfidence: 0.5
    });
    
    // Face Mesh
    faceMesh = new FaceMesh({
        locateFile: (file) => {
            return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`;
        }
    });
    
    faceMesh.setOptions({
        maxNumFaces: 1,
        refineLandmarks: true,
        minDetectionConfidence: 0.7,
        minTrackingConfidence: 0.5
    });
    
    // Pose
    pose = new Pose({
        locateFile: (file) => {
            return `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`;
        }
    });
    
    pose.setOptions({
        modelComplexity: 1,
        smoothLandmarks: true,
        minDetectionConfidence: 0.7,
        minTrackingConfidence: 0.5
    });
}

// カメラ初期化
async function initializeCamera(videoElement) {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: 640,
                height: 480,
                facingMode: 'user'
            }
        });
        videoElement.srcObject = stream;
        return new Promise((resolve) => {
            videoElement.onloadedmetadata = () => {
                resolve(stream);
            };
        });
    } catch (error) {
        console.error('カメラの初期化に失敗しました:', error);
        alert('カメラへのアクセスが許可されていません。');
        throw error;
    }
}

// 特徴量抽出
function extractFeatures(handResults, faceResults, poseResults) {
    const features = [];
    
    // 手の特徴量
    if (handResults.multiHandLandmarks) {
        for (let hand of handResults.multiHandLandmarks) {
            for (let landmark of hand) {
                features.push(landmark.x, landmark.y, landmark.z);
            }
        }
    }
    
    // 手が検出されない場合はゼロで埋める
    while (features.length < 42) { // 21 landmarks * 2 hands * 3 coordinates
        features.push(0, 0, 0);
    }
    
    // 顔の特徴量
    if (faceResults.multiFaceLandmarks) {
        for (let landmark of faceResults.multiFaceLandmarks[0]) {
            features.push(landmark.x, landmark.y, landmark.z);
        }
    } else {
        // 顔が検出されない場合はゼロで埋める
        for (let i = 0; i < 468 * 3; i++) {
            features.push(0);
        }
    }
    
    // ポーズの特徴量
    if (poseResults.poseLandmarks) {
        for (let landmark of poseResults.poseLandmarks) {
            features.push(landmark.x, landmark.y, landmark.z);
        }
    } else {
        // ポーズが検出されない場合はゼロで埋める
        for (let i = 0; i < 33 * 3; i++) {
            features.push(0);
        }
    }
    
    return features;
}

// 認識モード開始
async function startRecognition() {
    if (isRecognitionActive) return;
    
    try {
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        
        await initializeMediaPipe();
        await initializeCamera(video);
        
        isRecognitionActive = true;
        document.getElementById('start-btn').disabled = true;
        document.getElementById('stop-btn').disabled = false;
        
        // 認識ループ
        async function recognitionLoop() {
            if (!isRecognitionActive) return;
            
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            
            // MediaPipe処理
            const handResults = await hands.send({image: video});
            const faceResults = await faceMesh.send({image: video});
            const poseResults = await pose.send({image: video});
            
            // 特徴量抽出
            const features = extractFeatures(handResults, faceResults, poseResults);
            
            // ここでモデル推論を行う（現在はダミー）
            const prediction = await predictSign(features);
            
            // 結果表示
            document.getElementById('prediction-text').textContent = prediction.label;
            document.getElementById('confidence').textContent = `信頼度: ${(prediction.confidence * 100).toFixed(1)}%`;
            
            // ランドマーク描画
            drawLandmarks(ctx, handResults, faceResults, poseResults);
            
            requestAnimationFrame(recognitionLoop);
        }
        
        recognitionLoop();
        
    } catch (error) {
        console.error('認識開始エラー:', error);
        isRecognitionActive = false;
    }
}

// 認識モード停止
function stopRecognition() {
    isRecognitionActive = false;
    document.getElementById('start-btn').disabled = false;
    document.getElementById('stop-btn').disabled = true;
    
    // カメラ停止
    const video = document.getElementById('video');
    if (video.srcObject) {
        video.srcObject.getTracks().forEach(track => track.stop());
    }
}

// データ収集開始
async function startDataCollection() {
    const signWord = document.getElementById('sign-word').value.trim();
    const description = document.getElementById('sign-description').value.trim();
    const samplesCount = parseInt(document.getElementById('samples-count').value);
    
    if (!signWord) {
        alert('手話の単語を入力してください。');
        return;
    }
    
    if (samplesCount < 1 || samplesCount > 50) {
        alert('サンプル数は1-50の間で設定してください。');
        return;
    }
    
    try {
        const video = document.getElementById('collection-video');
        const canvas = document.getElementById('collection-canvas');
        
        await initializeMediaPipe();
        await initializeCamera(video);
        
        isCollectionActive = true;
        currentSignWord = signWord;
        targetSamples = samplesCount;
        currentSample = 0;
        collectedData = [];
        
        document.getElementById('capture-btn').disabled = false;
        document.getElementById('save-btn').disabled = true;
        
        updateCollectionStatus();
        
    } catch (error) {
        console.error('データ収集開始エラー:', error);
    }
}

// サンプル撮影
async function captureSample() {
    if (!isCollectionActive || currentSample >= targetSamples) return;
    
    try {
        const video = document.getElementById('collection-video');
        
        // MediaPipe処理
        const handResults = await hands.send({image: video});
        const faceResults = await faceMesh.send({image: video});
        const poseResults = await pose.send({image: video});
        
        // 特徴量抽出
        const features = extractFeatures(handResults, faceResults, poseResults);
        
        // データ保存
        const sampleData = {
            timestamp: Date.now(),
            signWord: currentSignWord,
            features: features,
            handResults: handResults,
            faceResults: faceResults,
            poseResults: poseResults
        };
        
        collectedData.push(sampleData);
        currentSample++;
        
        updateCollectionStatus();
        
        if (currentSample >= targetSamples) {
            document.getElementById('capture-btn').disabled = true;
            document.getElementById('save-btn').disabled = false;
        }
        
    } catch (error) {
        console.error('サンプル撮影エラー:', error);
    }
}

// データ保存
async function saveData() {
    if (collectedData.length === 0) return;
    
    try {
        const saveBtn = document.getElementById('save-btn');
        saveBtn.innerHTML = '<span class="loading"></span> 保存中...';
        saveBtn.disabled = true;
        
        // GitHubにデータを送信
        const response = await fetch('https://api.github.com/repos/your-username/SignMVP/contents/web_data', {
            method: 'POST',
            headers: {
                'Authorization': 'token YOUR_GITHUB_TOKEN',
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: `Add sign language data: ${currentSignWord}`,
                content: btoa(JSON.stringify(collectedData)),
                branch: 'main'
            })
        });
        
        if (response.ok) {
            alert(`${currentSignWord}のデータが正常に保存されました！`);
            resetCollection();
        } else {
            throw new Error('データ保存に失敗しました');
        }
        
    } catch (error) {
        console.error('データ保存エラー:', error);
        alert('データ保存に失敗しました。');
    } finally {
        const saveBtn = document.getElementById('save-btn');
        saveBtn.innerHTML = 'データ保存';
        saveBtn.disabled = false;
    }
}

// 収集状況更新
function updateCollectionStatus() {
    document.getElementById('collection-text').textContent = 
        `収集中: ${currentSignWord}`;
    document.getElementById('collection-progress').textContent = 
        `${currentSample} / ${targetSamples}`;
}

// 収集リセット
function resetCollection() {
    isCollectionActive = false;
    collectedData = [];
    currentSignWord = '';
    targetSamples = 0;
    currentSample = 0;
    
    document.getElementById('capture-btn').disabled = true;
    document.getElementById('save-btn').disabled = true;
    document.getElementById('collection-text').textContent = '準備完了';
    document.getElementById('collection-progress').textContent = '0 / 0';
    
    // カメラ停止
    const video = document.getElementById('collection-video');
    if (video.srcObject) {
        video.srcObject.getTracks().forEach(track => track.stop());
    }
}

// ランドマーク描画
function drawLandmarks(ctx, handResults, faceResults, poseResults) {
    // 手のランドマーク
    if (handResults.multiHandLandmarks) {
        for (let landmarks of handResults.multiHandLandmarks) {
            drawConnectors(ctx, landmarks, HAND_CONNECTIONS, {color: '#00FF00', lineWidth: 2});
            drawLandmarks(ctx, landmarks, {color: '#FF0000', lineWidth: 1, radius: 3});
        }
    }
    
    // 顔のランドマーク
    if (faceResults.multiFaceLandmarks) {
        for (let landmarks of faceResults.multiFaceLandmarks) {
            drawConnectors(ctx, landmarks, FACEMESH_TESSELATION, {color: '#C0C0C070', lineWidth: 1});
        }
    }
    
    // ポーズのランドマーク
    if (poseResults.poseLandmarks) {
        drawConnectors(ctx, poseResults.poseLandmarks, POSE_CONNECTIONS, {color: '#00FFFF', lineWidth: 2});
        drawLandmarks(ctx, poseResults.poseLandmarks, {color: '#FF00FF', lineWidth: 1, radius: 3});
    }
}

// ダミー予測関数（実際のモデルに置き換える）
async function predictSign(features) {
    // 実際の実装では、ここでTensorFlow.jsモデルを使用
    const dummyPredictions = [
        {label: 'こんにちは', confidence: 0.85},
        {label: 'ありがとう', confidence: 0.72},
        {label: 'さようなら', confidence: 0.68},
        {label: 'other', confidence: 0.45}
    ];
    
    // ランダムな予測を返す（デモ用）
    const randomIndex = Math.floor(Math.random() * dummyPredictions.length);
    return dummyPredictions[randomIndex];
}

// 初期化
document.addEventListener('DOMContentLoaded', function() {
    console.log('SignMVP Web App が読み込まれました');
});
