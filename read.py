# ...

def read_temperature_data(filename, keys):
    with open(filename, 'rb') as file:
        # 最初の 1KB を読み込む
        meta = file.read(1024).decode().split("\x00", 1)[0]

        # メタデータを解析
        metadata = {}
        for line in meta.split("\n"):
            if ':' in line:
                key, value = line.split(":", 1)
                metadata[key] = value.strip()

        # 指定されたすべての key の値を取得
        for key in keys:
            if key not in metadata:
                raise ValueError(f"キー '{key}' が見つかりません。")
            print(f"{key}: {metadata[key]}")

        # 残りのデータを読み込む
        data = np.fromfile(file, dtype=np.float32)

    return metadata, data

# ...

# 複数のキーを指定してデータを読み込む
keys = ['timestamp', 'deviceID', 'sensorType']
metadata, temperature_data = read_temperature_data(filename, keys)
print("温度データ:", temperature_data)
