import tkinter as tk
import tkinter.filedialog as fd
import tkinter.messagebox as mb
import PIL.Image, PIL.ImageTk, PIL.ImageDraw
import sklearn.datasets
import sklearn.svm
import sklearn.preprocessing
import numpy as np
import sys, os

# 学習データファイルの保存先をスクリプトと同じフォルダに設定
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(__file__)
DATA_FILE = os.path.join(BASE_DIR, "data.npz")  # 学習データファイルのパス

class DigitLearnerApp:

    def __init__(self, root):
        # ウィンドウ設定
        self.root = root
        self.root.title("手書き数字認識アプリ(ver1.0)")
        self.root.geometry("500x700")

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

        # 手書きキャンバスの作成と描画設定
        self.canvas = tk.Canvas(root, bg="white", width=280, height=280)
        self.canvas.pack(pady=5)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.image_for_draw = PIL.Image.new("L", (280, 280), "white")
        self.draw_obj = PIL.ImageDraw.Draw(self.image_for_draw)

        # 認識・クリアボタンの作成
        btn_frame = tk.Frame(root)
        btn_frame.pack()
        self.predict_btn = tk.Button(btn_frame, text="認識", command=self.predict_canvas, width=10)
        self.clear_btn = tk.Button(btn_frame, text="クリア", command=self.clear_canvas, width=10)
        self.predict_btn.pack(side=tk.LEFT, padx=5, pady=5)
        self.clear_btn.pack(side=tk.LEFT, padx=5, pady=5)

        # 認識結果表示ラベル
        self.text_label = tk.Label(root, text="手書きの数字を認識します！", font=("Helvetica", 14))
        self.text_label.pack(pady=5)

        # 正解入力欄と学習ボタン
        correct_frame = tk.Frame(root)
        correct_frame.pack()
        self.correct_entry = tk.Entry(correct_frame, width=5, font=("Helvetica", 14))
        self.add_btn = tk.Button(correct_frame, text="正解で学習", command=self.add_training_data)
        tk.Label(correct_frame, text="正解:").pack(side=tk.LEFT)
        self.correct_entry.pack(side=tk.LEFT)
        self.add_btn.pack(side=tk.LEFT)

        # 画像ファイル読み込みボタン
        self.load_btn = tk.Button(root, text="画像ファイルを開く", command=self.open_file, width=25)
        self.load_btn.pack(pady=5)

        # 読み込んだ画像の表示ラベル
        self.image_label = tk.Label(root)
        self.image_label.pack()

    # 保存済み学習データの読み込み
    def load_data(self):
        data = np.load(DATA_FILE)
        self.X = data["X"]
        self.y = data["y"]
        print(f"保存データを読み込みました（{len(self.y)}件）")

    # 学習データの保存
    def save_data(self):
        np.savez(DATA_FILE, X=self.X, y=self.y)
        print(f"データを保存しました（{len(self.y)}件）")

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
        self.text_label.configure(text=f"この画像は『{pred}』です！（確信度: {confidence:.1f}%）")
        return pred

    # キャンバス上の手書き数字を認識
    def predict_canvas(self):
        self.current_data = self.image_to_data(self.image_for_draw)
        self.current_pred = self.predict_digits(self.current_data)

    # 入力された正解ラベルでモデルを再学習
    def add_training_data(self):
        label = self.correct_entry.get()
        if label.isdigit():
            label = int(label)
            self.X = np.vstack([self.X, self.scaler.transform([self.current_data])])
            self.y = np.append(self.y, label)
            self.model = self.train_model()
            self.save_data()
            mb.showinfo("学習完了", f"正解『{label}』でモデルを更新しました！")
        else:
            mb.showerror("エラー", "0〜9の数字を入力してください")

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
        fpath = fd.askopenfilename(filetypes=[("画像ファイル", "*.png;*.jpg;*.jpeg;*.bmp")])
        if fpath:
            img = PIL.Image.open(fpath)
            dispImage = PIL.ImageTk.PhotoImage(img.resize((280, 280)))
            self.image_label.configure(image=dispImage)
            self.image_label.image = dispImage
            data = self.image_to_data(img)
            self.current_data = data
            self.current_pred = self.predict_digits(data)

# アプリケーションを起動
if __name__ == "__main__":
    root = tk.Tk()
    app = DigitLearnerApp(root)
    root.mainloop()
