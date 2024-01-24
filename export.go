// ...

// writeToFile は、データをファイルに書き込む関数
func writeToFile(port string, data []float32) {
	// 現在時刻を取得
	now := time.Now()
	timestamp := now.Format("20060102-150405")
	headerTimestamp := now.Format(time.RFC3339Nano)

	// ファイル名の生成（ポート名と同期時刻）
	filename := fmt.Sprintf("%s_%s.bin", port, timestamp)

	// ファイルを開く
	file, err := os.Create(filename)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	// メタデータ用の 1KB スペースを作成
	meta := make([]byte, 1024)

	// ヘッダーに key:value 形式のタイムスタンプを書き込む
	deviceID := "12345"
	sensorType := "thermal"

	// ヘッダーに複数の key:value を書き込む
	header := fmt.Sprintf("timestamp:%s\ndeviceID:%s\nsensorType:%s", headerTimestamp, deviceID, sensorType)
	copy(meta, []byte(header))
	// メタデータをファイルに書き込む
	_, err = file.Write(meta)
	if err != nil {
		panic(err)
	}

	// データをバイナリ形式で書き込む
	for _, value := range data {
		err = binary.Write(file, binary.LittleEndian, value)
		if err != nil {
			panic(err)
		}
	}
}

// ...
