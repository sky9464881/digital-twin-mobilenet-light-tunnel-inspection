**파일 구성**

1. **model\_train.py : 모델 훈련 -> best.py 뽑는 용도**

**-> checkpoints에 best.pth 저장**



**2. model\_export.py : 추출용(?) 모델**

**3. evaluate.py : 모델 평가 -> confusion matrix, recall, precision**

**eval\_results 에 지표 저장**



**4. export\_detect\_net.py : model\_export와 best.pth를 onnx로 변경**







**## 1\\) 데이터셋 구조**



**```text**

**YOUR\\\_DATASET/**

&#x20; **JPEGImages/**

&#x20;   **IMG\\\_0001.jpg**

&#x20;   **IMG\\\_0002.jpg**

&#x20; **Annotations/**

&#x20;   **IMG\\\_0001.xml**

&#x20;   **IMG\\\_0002.xml**

&#x20; **ImageSets/**

&#x20;   **Main/**

&#x20;     **train.txt**

&#x20;     **val.txt**

&#x20;     **test.txt**

&#x20; **labels.txt**

**```**



**### labels.txt 예시**



**```text**

**BACKGROUND**

**scratch**

**dent**

**smash**

**stain**

**```**





**##** 모델 실행

python model\_train.py  --data-root dataset --epochs 50 --batch-size 4 --img-size 320

&#x20;        (파일이름)                  (폴더)



\## onnx 변환

python export\_detectnet\_onnx.py  --checkpoint checkpoints/epoch\_50.pth  --model-py model\_export.py  --labels dataset/labels.txt  --output detectNet.onnx

&#x20;                  (파일 이름)                                    (폴더)/(이름).pth                            (모델).py                             (레이블).txt                   (출력).onnx



\## 평가

python evaluate.py --model-py model\_train.py --checkpoints checkpoints/epoch\_50.pth --data-root dataset --split test --score-thresh 0.19 --iou-match-thresh 0.50 --output-dir eval\_results





\# model.py의 output 구조

{

&#x20;   "model\_state\_dict": model.state\_dict(),

&#x20;   "classes": \["scratch", "dent", "dirt", "smash"]

&#x20;   "img\_size": ..., \[int = 320]

&#x20;   "normalization": {
"range": \[0.0, 1.0],

&#x09;	"mean": \[0.5, 0.5, 0.5],

&#x09;	"std": \[0.5, 0.5, 0.5]

&#x09;},

&#x20;   "detectnet\_export": {

&#x09;	"input\_name": "input\_0",

&#x09;	"scores\_name": "scores",

&#x09;	"boxes\_name": "boxes"

&#x09;}

}

