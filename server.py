# pip install grpcio-tools
# python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. service.proto
from concurrent import futures
import grpc
import service_pb2
import service_pb2_grpc
import grpc_health.v1.health_pb2 as health_pb2
import grpc_health.v1.health_pb2_grpc as health_pb2_grpc

class MyService(service_pb2_grpc.MyServiceServicer):
    def GetData(self, request, context):
        sample_data = [1, 2, 3, 4, 5]
        return service_pb2.ShortArrayResponse(data=sample_data)

class HealthServicer(health_pb2_grpc.HealthServicer):
    def Check(self, request, context):
        return health_pb2.HealthCheckResponse(status=health_pb2.HealthCheckResponse.SERVING)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    service_pb2_grpc.add_MyServiceServicer_to_server(MyService(), server)
    health_pb2_grpc.add_HealthServicer_to_server(HealthServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
