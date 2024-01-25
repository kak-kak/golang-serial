package main

import (
    "bufio"
    "fmt"
    "os"
    "strconv"
)

func main() {
    // ファイルを開く（存在しない場合は作成）
    file, err := os.OpenFile("output.txt", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
    if err != nil {
        fmt.Println("ファイルのオープンに失敗しました:", err)
        return
    }
    defer file.Close()

    writer := bufio.NewWriter(file)

    // 例として10回のループ
    for i := 0; i < 10; i++ {
        // 配列を生成
        array := []int{i, i * 2, i * 3}

        // 配列の各要素を文字列に変換し、ファイルに書き込む
        for _, value := range array {
            _, err := writer.WriteString(strconv.Itoa(value) + "\n")
            if err != nil {
                fmt.Println("ファイルへの書き込みに失敗しました:", err)
                return
            }
        }
    }

    // バッファの内容をフラッシュ
    if err := writer.Flush(); err != nil {
        fmt.Println("ファイルへのフラッシュに失敗しました:", err)
    }
}
