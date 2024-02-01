package main

import (
	"fmt"
	"gocv.io/x/gocv"
	"sync"
	"time"
)

type MultiCameraCapture struct {
	cameras     []*gocv.VideoCapture
	isOpen      []bool
	mutex       sync.Mutex
}

func NewMultiCameraCapture(numCameras int) (*MultiCameraCapture, error) {
	mcc := &MultiCameraCapture{
		cameras: make([]*gocv.VideoCapture, numCameras),
		isOpen:  make([]bool, numCameras),
	}

	for i := 0; i < numCameras; i++ {
		mcc.cameras[i] = gocv.NewVideoCapture(i)
		if !mcc.cameras[i].IsOpened() {
			return nil, fmt.Errorf("カメラ %d のオープンに失敗しました", i)
		}
		mcc.isOpen[i] = true
	}

	return mcc, nil
}

func (mcc *MultiCameraCapture) CloseAll() {
	mcc.mutex.Lock()
	defer mcc.mutex.Unlock()

	for i, camera := range mcc.cameras {
		if mcc.isOpen[i] {
			camera.Close()
			mcc.isOpen[i] = false
		}
	}
}

func (mcc *MultiCameraCapture) RestartCamera(cameraIndex int) error {
	mcc.mutex.Lock()
	defer mcc.mutex.Unlock()

	if cameraIndex < 0 || cameraIndex >= len(mcc.cameras) {
		return fmt.Errorf("無効なカメラインデックスです")
	}

	if mcc.isOpen[cameraIndex] {
		mcc.cameras[cameraIndex].Close()
		mcc.isOpen[cameraIndex] = false
	}

	mcc.cameras[cameraIndex] = gocv.NewVideoCapture(cameraIndex)
	if !mcc.cameras[cameraIndex].IsOpened() {
		return fmt.Errorf("カメラ %d のオープンに失敗しました", cameraIndex)
	}

	mcc.isOpen[cameraIndex] = true
	return nil
}

func (mcc *MultiCameraCapture) Read(cameraIndex int) (*gocv.Mat, error) {
	mcc.mutex.Lock()
	defer mcc.mutex.Unlock()

	if cameraIndex < 0 || cameraIndex >= len(mcc.cameras) {
		return nil, fmt.Errorf("無効なカメラインデックスです")
	}

	if !mcc.isOpen[cameraIndex] {
		return nil, fmt.Errorf("カメラ %d はオープンされていません", cameraIndex)
	}

	mat := gocv.NewMat()
	if ok := mcc.cameras[cameraIndex].Read(&mat); !ok {
		// リードに失敗した場合、リスタートを試みる
		fmt.Printf("カメラ %d のリードに失敗しました。リスタートを試みます...\n", cameraIndex)
		if err := mcc.RestartCamera(cameraIndex); err != nil {
			return nil, fmt.Errorf("カメラ %d のリスタートに失敗しました: %v", cameraIndex, err)
		}

		// リスタート後に再度リードを試みる
		if ok := mcc.cameras[cameraIndex].Read(&mat); !ok {
			return nil, fmt.Errorf("カメラ %d のリードに再び失敗しました", cameraIndex)
		}
	}

	return &mat, nil
}

func main() {
	// 例として2つのカメラを使用する場合
	numCameras := 2
	mcc, err := NewMultiCameraCapture(numCameras)
	if err != nil {
		fmt.Println("MultiCameraCaptureの作成エラー:", err)
		return
	}
	defer mcc.CloseAll()

	for i := 0; i < 10; i++ {
		mat, err := mcc.Read(0)
		if err != nil {
			fmt.Printf("リードエラー: %v\n", err)
			time.Sleep(1 * time.Second) // 失敗したら待機してから再試行するなどの処理を追加できます
			continue
		}

		// matを使用した処理を追加

		mat.Close()
	}
}
