package main

import (
    "encoding/csv"
    "fmt"
    "os"
    "time"
)

func main() {
    // CSVファイルを作成
    file, err := os.Create("output.csv")
    if err != nil {
        fmt.Println("ファイルの作成に失敗しました:", err)
        return
    }
    defer file.Close()

    // CSVライターを作成
    writer := csv.NewWriter(file)
    defer writer.Flush()

    // 現在のタイムスタンプを取得
    timestamp := time.Now().Format(time.RFC3339)

    // サンプルの配列データ
    data := []string{"Apple", "Banana", "Cherry"}

    // CSVにタイムスタンプとデータを書き込む
    record := append([]string{timestamp}, data...)
    err = writer.Write(record)
    if err != nil {
        fmt.Println("書き込みエラー:", err)
        return
    }

    fmt.Println("CSVファイルへの書き込みが完了しました。")
}
