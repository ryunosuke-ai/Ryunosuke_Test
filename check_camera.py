import cv2

def list_cameras():
    print("=== カメラ番号の確認 ===")
    # 0番から4番まで順番にテストしてみる
    for index in range(5):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"✅ カメラ番号 {index}: 接続成功")
            ret, frame = cap.read()
            if ret:
                print(f"   --> 映像取得OK ({frame.shape[1]}x{frame.shape[0]})")
            else:
                print("   --> 接続できたけど映像が取れません")
            cap.release()
        else:
            print(f"❌ カメラ番号 {index}: 接続失敗")

if __name__ == "__main__":
    list_cameras()