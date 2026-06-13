import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import sys

# verify.py의 로직을 사용하기 위해 import
import verify

class MathOCR_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("손글씨 수식 인식 및 검증기")
        self.root.geometry("1000x700")
        
        # 상단 제어부
        self.top_frame = tk.Frame(root, pady=10)
        self.top_frame.pack(side=tk.TOP, fill=tk.X)
        
        self.btn_load = tk.Button(self.top_frame, text="수식 이미지 불러오기", command=self.load_image, padx=20, pady=5)
        self.btn_load.pack()
        
        # 메인 컨텐츠 영역 (스크롤 가능하게 구성)
        self.canvas = tk.Canvas(root, bg="white")
        self.scrollbar = tk.Scrollbar(root, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg="white")

        # 윈도우 크기 변경 시 스크롤 영역 업데이트 및 중앙 정렬 유지
        self.canvas_window = self.canvas.create_window((500, 0), window=self.scrollable_frame, anchor="n")

        def on_canvas_configure(e):
            # 캔버스 너비에 맞춰 윈도우 위치 조정 (중앙 정렬)
            self.canvas.itemconfig(self.canvas_window, width=e.width)
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

        self.canvas.bind("<Configure>", on_canvas_configure)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # 위젯들 담을 리스트
        self.char_labels = []
        self.original_img_label = None

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not file_path:
            return
            
        # 기존 화면 초기화
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
            
        try:
            # 1. 원본 이미지 표시
            title_orig = tk.Label(self.scrollable_frame, text="[ 입력 이미지 ]", font=("Arial", 14, "bold"), pady=15, bg="white")
            title_orig.pack(anchor="center")
            
            orig_img = Image.open(file_path)
            # 화면 크기에 맞춰 리사이징 (비율 유지)
            display_w = 700 # 너비를 조금 줄여 여유 공간 확보
            w, h = orig_img.size
            ratio = display_w / w
            display_h = int(h * ratio)
            orig_display = orig_img.resize((display_w, display_h), Image.LANCZOS)
            
            self.tk_orig = ImageTk.PhotoImage(orig_display)
            self.original_img_label = tk.Label(self.scrollable_frame, image=self.tk_orig, bg="white")
            self.original_img_label.pack(anchor="center")
            
            # 2. 기호 분할 및 인식 시작
            status_label = tk.Label(self.scrollable_frame, text="수식 분석 중...", font=("Arial", 12), pady=20, bg="white")
            status_label.pack(anchor="center")
            self.root.update()
            
            chars = verify.segment_characters(file_path)
            if not chars:
                messagebox.showerror("오류", "이미지에서 기호를 분할하지 못했습니다.")
                status_label.destroy()
                return
                
            # 기호들을 보여줄 프레임
            title_chars = tk.Label(self.scrollable_frame, text="[ 기호별 인식 결과 ]", font=("Arial", 14, "bold"), pady=15, bg="white")
            title_chars.pack(anchor="center")
            
            # 기호 컨테이너 자체를 중앙에 배치하기 위한 프레임
            char_outer_container = tk.Frame(self.scrollable_frame, bg="white")
            char_outer_container.pack(anchor="center")
            
            char_container = tk.Frame(char_outer_container, pady=10, bg="white")
            char_container.pack()
            
            symbols = []
            self.tk_chars = [] 
            
            for i, char_img in enumerate(chars):
                symbol = verify.predict_symbol(char_img, i)
                symbols.append(symbol)
                
                unit_frame = tk.Frame(char_container, borderwidth=2, relief="groove", padx=8, pady=8, bg="white")
                unit_frame.pack(side=tk.LEFT, padx=8)
                
                processed_arr = verify.preprocess_image(char_img) * 255.0
                disp_char = Image.fromarray(processed_arr.astype(np.uint8)).resize((70, 70), Image.NEAREST)
                
                tk_char = ImageTk.PhotoImage(disp_char)
                self.tk_chars.append(tk_char)
                
                img_lbl = tk.Label(unit_frame, image=tk_char, bg="white")
                img_lbl.pack()
                
                txt_lbl = tk.Label(unit_frame, text=f"{symbol}", font=("Arial", 14, "bold"), bg="white", pady=5)
                txt_lbl.pack()
            
            # 3. 최종 수식 검증 결과
            status_label.destroy()
            
            res_container = tk.Frame(self.scrollable_frame, bg="white", pady=30)
            res_container.pack(fill=tk.X)

            res_frame = tk.Frame(res_container, pady=25, bg="#f8f9fa", padx=30, relief="ridge", borderwidth=1)
            res_frame.pack(anchor="center", padx=50)
            
            clean_symbols = [s for s in symbols if s is not None]
            idx = 0
            while idx < len(clean_symbols) - 1:
                if clean_symbols[idx] == "=" and clean_symbols[idx + 1] == "=":
                    clean_symbols.pop(idx + 1)
                else:
                    idx += 1
            
            eq_str = " ".join(str(s) for s in clean_symbols)
            tk.Label(res_frame, text=f"인식된 수식:  {eq_str}", font=("Arial", 18, "bold"), bg="#f8f9fa").pack(pady=10)
            
            if "=" not in clean_symbols:
                result_text = "● 등호(=) 누락 ●"
                result_color = "#95a5a6"
                bg_color = "#f2f3f4"
            else:
                eq_idx = clean_symbols.index("=")
                left = clean_symbols[:eq_idx]
                right = clean_symbols[eq_idx + 1 :]
                
                if not left or not right:
                    result_text = "● 수식 불완전 ●"
                    result_color = "#f39c12"
                    bg_color = "#fef5e7"
                else:
                    left_expr = "".join(str(s) for s in left)
                    right_expr = "".join(str(s) for s in right)
                    
                    try:
                        left_val = eval(left_expr)
                        right_val = eval(right_expr)
                        
                        detail_text = f"좌변: {left_expr} = {left_val}  |  우변: {right_expr} = {right_val}"
                        tk.Label(res_frame, text=detail_text, font=("Arial", 13), bg="#f8f9fa", pady=15).pack()
                        
                        if abs(left_val - right_val) < 1e-7:
                            result_text = "● 참 (TRUE) ●"
                            result_color = "#27ae60"
                            bg_color = "#eafaf1"
                        else:
                            result_text = "○ 거짓 (FALSE) ○"
                            result_color = "#c0392b"
                            bg_color = "#f9ebea"
                    except Exception as e:
                        result_text = "● 계산 오류 ●"
                        result_color = "#8e44ad"
                        bg_color = "#f5eef8"
            
            final_res_frame = tk.Frame(res_frame, bg=bg_color, highlightbackground=result_color, 
                                      highlightthickness=4, bd=0, pady=25, padx=40)
            final_res_frame.pack(pady=15)
            
            tk.Label(final_res_frame, text=result_text, font=("Arial", 36, "bold"), 
                     fg=result_color, bg=bg_color).pack()

            
        except Exception as e:
            messagebox.showerror("오류", f"이미지 처리 중 오류가 발생했습니다: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    import numpy as np # GUI 내부에서 np 사용을 위해 추가
    root = tk.Tk()
    app = MathOCR_GUI(root)
    root.mainloop()
