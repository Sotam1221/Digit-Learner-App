import tkinter as tk
import tkinter.filedialog as fd
import tkinter.messagebox as mb
import PIL.Image, PIL.ImageTk, PIL.ImageDraw
import sklearn.datasets
import sklearn.svm
import sklearn.preprocessing
import numpy as np
import sys, os, csv
from datetime import datetime

# 学習データファイルの保存先をスクリプトと同じフォルダに設定
if getattr(sys, 'frozen', False):  
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(__file__)
DATA_FILE = os.path.join(BASE_DIR, "data.npz")

class DigitLearnerApp:

    def __init__(self, root):
        # ウィンドウ設定
        self.root = root
        self.root.title("Digit Learner App(ver2.1)")
        self.root.geometry("500x800")

        # 初期データセットをロードしスケーリング
        self.digits = sklearn.datasets.load_digits()
        self.scaler = sklearn.preprocessing.MinMaxScaler()
        self.X = self.scaler.fit_transform(self.digits.data)
        self.y = self.digits.target

        # 保存済みデータがあれば読み込み
        if os.path.exists(DATA_FILE):
            self.load_data()

        # SVMモデルの学習
        self.model = self.train_model()

        # 履歴データの初期化
        self.history = []
        self.history_count = 1

        # 手書きキャンバスの作成と描画設定
        self.canvas = tk.Canvas(root, bg="white", width=280, height=280)
        self.canvas.pack(pady=5)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.image_for_draw = PIL.Image.new("L", (280, 280), "white")
        self.draw_obj = PIL.ImageDraw.Draw(self.image_for_draw)

        # 認識・クリアボタンの作成
        btn_frame = tk.Frame(root)
        btn_frame.pack()
        self.predict_btn = tk.Button(btn_frame, text="Recognize", command=self.predict_canvas, width=10)
        self.clear_btn = tk.Button(btn_frame, text="Clear", command=self.clear_canvas, width=10)
        self.predict_btn.pack(side=tk.LEFT, padx=5, pady=5)
        self.clear_btn.pack(side=tk.LEFT, padx=5, pady=5)

        # 認識結果表示ラベル
        self.text_label = tk.Label(root, text="Draw a number to recognize!", font=("Helvetica", 14))
        self.text_label.pack(pady=5)

        # 正解入力欄と学習ボタン
        correct_frame = tk.Frame(root)
        correct_frame.pack()
        self.correct_entry = tk.Entry(correct_frame, width=5, font=("Helvetica", 14))
        self.add_btn = tk.Button(correct_frame, text="Learn", command=self.add_training_data)
        tk.Label(correct_frame, text="Correct:").pack(side=tk.LEFT)
        self.correct_entry.pack(side=tk.LEFT)
        self.add_btn.pack(side=tk.LEFT)

        # 学習データ初期化ボタンの作成
        reset_frame = tk.Frame(root)
        reset_frame.pack(pady=10)
        self.reset_btn = tk.Button(reset_frame, text="Reset Training Data", command=self.reset_data, width=20)
        self.reset_btn.pack()

        # 履歴CSV出力ボタンの作成
        export_frame = tk.Frame(root)
        export_frame.pack(pady=10)
        self.export_btn = tk.Button(export_frame, text="Export History (CSV)", command=self.export_history, width=20)
        self.export_btn.pack()

        # 画像ファイル読み込みボタン
        self.load_btn = tk.Button(root, text="Open Image File", command=self.open_file, width=25)
        self.load_btn.pack(pady=5)

        # 読み込んだ画像の表示ラベル
        self.image_label = tk.Label(root)
        self.image_label.pack()

    # 保存済み学習データの読み込み
    def load_data(self):
        data = np.load(DATA_FILE)
        self.X = data["X"]
        self.y = data["y"]
        print(f"Loaded data: {len(self.y)} samples")

    # 学習データの保存
    def save_data(self):
        np.savez(DATA_FILE, X=self.X, y=self.y)
        print(f"Saved data: {len(self.y)} samples")

    # 学習データを初期状態に戻す
    def reset_data(self):
        if mb.askyesno("Confirm", "Do you really want to reset training data?"):
            if os.path.exists(DATA_FILE):
                os.remove(DATA_FILE)
            self.X = self.scaler.fit_transform(self.digits.data)
            self.y = self.digits.target
            self.model = self.train_model()
            mb.showinfo("Reset", "Training data has been reset.")

    # 学習履歴をCSV出力
    def export_history(self):
        if not self.history:
            mb.showinfo("Info", "No history data available.")
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(BASE_DIR, f"history_{timestamp}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["No", "datetime", "predicted", "correct", "confidence(%)", "is_correct(1/0)"])
            writer.writerows(self.history)
        mb.showinfo("Done", f"History exported:\n{csv_path}")
        try:
            os.startfile(csv_path)
        except Exception as e:
            print(f"Could not open CSV: {e}")

    # SVMモデルの学習処理
    def train_model(self):
        clf = sklearn.svm.SVC(probability=True, C=10, gamma=0.005)
        clf.fit(self.X, self.y)
        return clf

    # 画像を8x8の数値配列に変換
    def image_to_data(self, pil_img):
        grayImage = pil_img.convert("L")
        grayImage = grayImage.resize((8, 8), PIL.Image.Resampling.LANCZOS)
        npImage = np.asarray(grayImage, dtype=float)
        npImage = 16 - np.floor(17 * npImage / 256)
        return npImage.flatten()

    # モデルによる数字の予測
    def predict_digits(self, data):
        data_scaled = self.scaler.transform([data])
        pred = self.model.predict(data_scaled)[0]
        prob = self.model.predict_proba(data_scaled)[0]
        confidence = np.max(prob) * 100
        self.text_label.configure(text=f"Predicted: {pred} (Confidence: {confidence:.1f}%)")
        return pred, confidence

    # キャンバス上の手書き数字を認識
    def predict_canvas(self):
        self.current_data = self.image_to_data(self.image_for_draw)
        self.current_pred, self.current_confidence = self.predict_digits(self.current_data)

    # 入力された正解ラベルでモデルを再学習し履歴に記録
    def add_training_data(self):
        label = self.correct_entry.get()
        if label.isdigit():
            label = int(label)
            correct_flag = 1 if self.current_pred == label else 0
            self.history.append([
                self.history_count,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                self.current_pred,
                label,
                f"{self.current_confidence:.1f}",
                correct_flag
            ])
            self.history_count += 1
            self.X = np.vstack([self.X, self.scaler.transform([self.current_data])])
            self.y = np.append(self.y, label)
            self.model = self.train_model()
            self.save_data()
            mb.showinfo("Learned", f"Model updated with correct label: {label}")
            self.correct_entry.delete(0, tk.END)
            self.clear_canvas()
        else:
            mb.showerror("Error", "Please enter a number between 0 and 9.")

    # マウスドラッグでキャンバスに描画
    def draw(self, event):
        x, y = event.x, event.y
        r = 10
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="black", width=0)
        self.draw_obj.ellipse([x-r, y-r, x+r, y+r], fill="black")

    # キャンバスを初期化
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image_for_draw = PIL.Image.new("L", (280, 280), "white")
        self.draw_obj = PIL.ImageDraw.Draw(self.image_for_draw)

    # 画像ファイルを読み込み認識
    def open_file(self):
        fpath = fd.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if fpath:
            img = PIL.Image.open(fpath)
            dispImage = PIL.ImageTk.PhotoImage(img.resize((280, 280)))
            self.image_label.configure(image=dispImage)
            self.image_label.image = dispImage
            data = self.image_to_data(img)
            self.current_data = data
            self.current_pred, self.current_confidence = self.predict_digits(data)

# アプリケーションを起動
if __name__ == "__main__":
    root = tk.Tk()
    app = DigitLearnerApp(root)
    root.mainloop()
