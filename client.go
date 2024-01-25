// go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
// go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
// protoc --go_out=. --go_opt=paths=source_relative --go-grpc_out=. --go-grpc_opt=paths=source_relative service.proto

package main

import (
    "context"
    "log"
    "time"

    "google.golang.org/grpc"
    pb "path/to/your/protobuf/package"
    healthpb "path/to/grpc/health/v1"
)

func main() {
    const address = "localhost:50051"

    conn, err := grpc.Dial(address, grpc.WithInsecure(), grpc.WithBlock())
    if err != nil {
        log.Fatalf("サーバーへの接続に失敗しました: %v", err)
    }
    defer conn.Close()

    // ヘルスチェック
    healthClient := healthpb.NewHealthClient(conn)
    healthResponse, err := healthClient.Check(context.Background(), &healthpb.HealthCheckRequest{})
    if err != nil {
        log.Fatalf("ヘルスチェックに失敗しました: %v", err)
    }
    log.Printf("ヘルスチェックの状態: %v", healthResponse.Status)

    // データ取得リクエスト
    c := pb.NewMyServiceClient(conn)
    ctx, cancel := context.WithTimeout(context.Background(), time.Second)
    defer cancel()
    r, err := c.GetData(ctx, &pb.EmptyRequest{})
    if err != nil {
        log.Fatalf("リクエストの実行に失敗しました: %v", err)
    }
    log.Printf("レスポンス: %v", r.GetData())
}
