import pandas as pd
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import os 
class DataFrameImageViewer:
    """
    An interactive viewer for images in a pandas DataFrame.
    Allows cycling through images that meet specific criteria,
    displaying the image, its mask, and the overlay.
    """
    def __init__(self, dataframe, image_col='image_path', mask_col='mask_path',predict_col='predict_path', info_cols=None):
        """
        Initializes the viewer.

        Args:
            dataframe (pd.DataFrame): The filtered DataFrame containing image paths.
            image_col (str): The name of the column with paths to the main images.
            mask_col (str): The name of the column with paths to the mask images.
            info_cols (list): A list of other column names to display as info.
        """
        if dataframe.empty:
            raise ValueError("The provided DataFrame is empty. No images to display.")
            
        self.df = dataframe
        self.image_col = image_col
        self.mask_col = mask_col
        self.info_cols = info_cols if info_cols else []
        self.predict_col = predict_col
        self.current_index = 0
        
        self.marked_data = []
        # --- Create the figure and axes ---
        self.fig, self.axes = plt.subplots(1, 4, figsize=(24, 7))
        self.fig.suptitle("Interactive Image Viewer", fontsize=16)
        self.fig.subplots_adjust(bottom=0.2)
        # --- Create Navigation Buttons ---
        ax_prev = plt.axes([0.5, 0.05, 0.1, 0.075])  # [left, bottom, width, height]
        ax_next = plt.axes([0.61, 0.05, 0.1, 0.075])
        ax_mark = plt.axes([0.72, 0.05, 0.1, 0.075])
        ax_unmark = plt.axes([0.83, 0.05, 0.1, 0.075])
        ax_export = plt.axes([0.94, 0.05, 0.1, 0.075])
        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_next = Button(ax_next, 'Next')
        self.btn_mark = Button(ax_mark, 'Mark')
        self.btn_unmark = Button(ax_unmark, 'Unmark')
        self.btn_export = Button(ax_export, 'Export Marked')

        # Connect buttons to callback functions
        self.btn_prev.on_clicked(self.prev_image)
        self.btn_next.on_clicked(self.next_image)
        self.btn_mark.on_clicked(self.mark_image)
        self.btn_unmark.on_clicked(self.unmark_image)
        self.btn_export.on_clicked(self.export_data)

        # --- Initial plot ---
        self.update_display()
    
    def _get_current_item_identifier(self):
        """Helper function to get the unique identifier for the current row."""
        if 'region' in self.df.columns and 'filename' in self.df.columns:
            row = self.df.iloc[self.current_index]
            return {'region': row['region'], 'filename': row['filename']}
        return None
    
    def mark_image(self, event):
        """Adds the current image's identifier to the marked list if not already present."""
        item = self._get_current_item_identifier()
        if item:
            if item not in self.marked_data:
                self.marked_data.append(item)
                print(f"Marked: {item['filename']}. Total marked: {len(self.marked_data)}")
                self.update_display() # Update display to show [MARKED] status
            else:
                print(f"Already marked: {item['filename']}")

    def unmark_image(self, event):
        """Removes the current image's identifier from the marked list if present."""
        item = self._get_current_item_identifier()
        if item:
            if item in self.marked_data:
                self.marked_data.remove(item)
                print(f"Unmarked: {item['filename']}. Total marked: {len(self.marked_data)}")
                self.update_display() # Update display to remove [MARKED] status
            else:
                print(f"Not marked yet: {item['filename']}")

    def export_data(self, event):
        """Exports the list of marked data to a CSV file."""
        if not self.marked_data:
            print("No data has been marked yet. Nothing to export.")
            return
        marked_df = pd.DataFrame(self.marked_data).drop_duplicates()
        output_filename = 'marked_images.csv'
        try:
            marked_df.to_csv(output_filename, index=False)
            print(f"Successfully exported {len(marked_df)} marked items to '{output_filename}'")
        except Exception as e:
            print(f"Error exporting file: {e}")
    
    def update_display(self):
        """Loads and displays images and updates the title with marked status."""
        for ax in self.axes:
            ax.clear()

        row = self.df.iloc[self.current_index]
        image_path, mask_path, predict_path = row[self.image_col], row[self.mask_col], row[self.predict_col]
        
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        predict = cv2.imread(predict_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            self.axes[0].text(0.5, 0.5, f"Image not found:\n{image_path}", ha='center', color='red')
            plt.draw()
            return
            
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_with_contours = image.copy()
        cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        overlay_rgb = cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB)

        self.axes[0].imshow(image_rgb); self.axes[0].set_title('Original Image')
        self.axes[1].imshow(mask, cmap='gray'); self.axes[1].set_title('Ground Truth Mask')
        self.axes[2].imshow(overlay_rgb); self.axes[2].set_title('Image with GT Contour')
        self.axes[3].imshow(predict, cmap='gray'); self.axes[3].set_title('Predicted Mask')
        
        for ax in self.axes:
            ax.axis('off')
            
        # --- IMPROVED: Display Info with persistent [MARKED] status ---
        info_text = f"Image {self.current_index + 1} of {len(self.df)}\n"
        
        # Check if the current item is marked and add a status indicator
        current_item = self._get_current_item_identifier()
        if current_item and current_item in self.marked_data:
            info_text = "[MARKED]\n" + info_text

        for col in self.info_cols:
            if col in row:
                info_text += f"{col}: {row.get(col, 'N/A')}\n"
        
        self.fig.suptitle(info_text, fontsize=12, x=0.05, y=0.98, ha='left', va='top')
        self.fig.canvas.draw_idle()



    def next_image(self, event):
        """Callback for 'Next' button."""
        self.current_index = (self.current_index + 1) % len(self.df)
        self.update_display()

    def prev_image(self, event):
        """Callback for 'Previous' button."""
        self.current_index = (self.current_index - 1) % len(self.df)
        self.update_display()


if __name__ == '__main__':
    try:
        df = pd.read_csv(r"F:\project\unetpp_find\data_slove_plot\region_data\evaluation_results_with_region.csv")
        dir_image_path = r"F:\project\unetpp_find\data\image\processed_chips\images"
        dir_mask_path = r"F:\project\unetpp_find\data\image\processed_chips\masks"
        dir_predict_path = r"F:\project\unetpp_find\data\image\processed_chips\predict"
        df['image_path'] = df.apply(lambda row: os.path.join(dir_image_path, row['region'],  row['filename']).replace('_mask', ''), axis=1)
        df['mask_path'] = df.apply(lambda row: os.path.join(dir_mask_path, row['region'],  row['filename']), axis=1)
        df['predict_path'] = df.apply(lambda row: os.path.join(dir_predict_path, row['region'],  "mask_" + row['filename']).replace("_mask",""), axis=1)
    except FileNotFoundError:
        print("错误：evaluation_results_with_region.csv 未找到。请检查文件路径。")
        # 创建一个假的DataFrame用于演示
        print("正在创建一个用于演示的假DataFrame。")
        data = {'image_path': ['placeholder.png']*5, 'mask_path': ['placeholder.png']*5, 'f1_score': [0, 0.8, 0, 0.9, 0.1], 'region': ['site1', 'site1', 'site2', 'site2', 'site3']}
        df = pd.DataFrame(data)

    condition = (df['f1_score']< 0.2)
    
    filtered_df = df[condition].reset_index(drop=True)

    # 3. 运行查看器
    if filtered_df.empty:
        print("没有找到符合条件的图片。")
    else:
        print(f"找到了 {len(filtered_df)} 张符合条件的图片。正在启动查看器...")
        # 定义要显示的额外信息列
        info_columns_to_display = ['f1_score', 'precision', 'recall', 'region',"filename"]
        
        # 确保这些列存在于DataFrame中
        valid_info_cols = [col for col in info_columns_to_display if col in filtered_df.columns]
        
        # 启动查看器 (请确保列名'image_path'和'mask_path'与您的DataFrame一致)
        viewer = DataFrameImageViewer(
            filtered_df, 
            image_col='image_path', 
            mask_col='mask_path', 
            info_cols=valid_info_cols
        )
        plt.show()