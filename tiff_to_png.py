import os
import rasterio
from rasterio.windows import Window
from rasterio.features import rasterize
import geopandas as gpd
import numpy as np
from PIL import Image
import glob
import traceback # For detailed error printing

def is_background_only(mask_chip, background_threshold=0.999, min_valid_pixels=10):
    """
    检测 mask_chip 是否主要是背景：
      1) 背景像素（==0）比例超过 background_threshold
      2) 或者 有效像素数（非 0）少于 min_valid_pixels
    """
    if mask_chip.size == 0:
        return True
    total_pixels = mask_chip.size
    background_pixels = np.sum(mask_chip == 0)
    valid_pixels = total_pixels - background_pixels
    
    background_ratio = background_pixels / total_pixels
    return (background_ratio >= background_threshold) or (valid_pixels < min_valid_pixels)

def rasterize_full_mask(tiff_path, shapefile_path,
                        mask_burn_attribute=None, default_burn_value=1):
    """
    将整个 shapefile 一次性 rasterize 成与 TIFF 同尺寸的掩膜数组。
    如果 Shapefile 和 TIFF 的 CRS 不同，Shapefile 会被自动重投影到 TIFF 的 CRS。
    """
    with rasterio.open(tiff_path) as src:
        tiff_crs = src.crs
        tiff_width = src.width
        tiff_height = src.height
        tiff_transform = src.transform

        if tiff_crs is None:
             raise ValueError(f"TIFF 文件 '{os.path.basename(tiff_path)}' 未定义 CRS。无法进行后续处理。")
        # Optional: Keep a check if you specifically need TIFFs to be in a certain CRS, e.g., EPSG:32644
        # For this modification, we'll assume the TIFF's CRS is the target CRS.
        # if tiff_crs.to_epsg() != 32644:
        #     print(f"警告: TIFF 文件 '{os.path.basename(tiff_path)}' 的 CRS 是 {tiff_crs}，而非预期的EPSG:32644。Shapefile将被投影到此CRS ({tiff_crs})。")


        # 1. 读取矢量
        gdf = gpd.read_file(shapefile_path)
        if gdf.empty:
            print(f"警告: Shapefile '{os.path.basename(shapefile_path)}' 为空。")
            return np.zeros((tiff_height, tiff_width), dtype=np.uint8), src.meta

        if gdf.crs is None:
            # Attempt to set a common default if CRS is missing, e.g., WGS84 (EPSG:4326)
            # This is a guess; it's always better if the shapefile has its CRS defined.
            print(f"警告: Shapefile '{os.path.basename(shapefile_path)}' 未定义 CRS。将尝试假设其为 EPSG:4326 (WGS84地理坐标系)。")
            try:
                gdf = gdf.set_crs("EPSG:4326", allow_override=True) # Use allow_override if you are sure about this assumption
            except Exception as e:
                raise ValueError(f"无法为 Shapefile '{os.path.basename(shapefile_path)}' 设置假定的EPSG:4326 CRS: {e}")

        # 2. 如果Shapefile的CRS与TIFF的CRS不同，则进行重投影
        if gdf.crs != tiff_crs:
            print(f"Shapefile CRS ({gdf.crs}) 与 TIFF CRS ({tiff_crs}) 不同。正在将Shapefile重投影到TIFF的CRS...")
            try:
                gdf = gdf.to_crs(tiff_crs)
                print(f"Shapefile已成功重投影到 {gdf.crs}.")
            except Exception as e:
                raise ValueError(f"Shapefile '{os.path.basename(shapefile_path)}' 重投影到 {tiff_crs} 失败: {e}")
        
        # 3. 把要素变成 shapes 列表
        shapes = []
        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty or not geom.is_valid:
                if geom and not geom.is_valid:
                    # Attempt to buffer by 0 to fix minor invalidities
                    geom_fixed = geom.buffer(0)
                    if geom_fixed is None or geom_fixed.is_empty or not geom_fixed.is_valid:
                        print(f"警告: Shapefile '{os.path.basename(shapefile_path)}' 中检测到无效几何图形（修复失败），已跳过。")
                        continue
                    else:
                        print(f"提示: Shapefile '{os.path.basename(shapefile_path)}' 中几何图形已通过buffer(0)修复。")
                        geom = geom_fixed
                else:
                    continue # geom is None or geom.is_empty
            
            burn_val = default_burn_value
            if mask_burn_attribute and mask_burn_attribute in row:
                v = row[mask_burn_attribute]
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    try:
                        burn_val = int(v)
                    except ValueError:
                        print(f"警告: 属性 '{mask_burn_attribute}' 的值 '{v}' 无法转换为整数，将使用默认值 {default_burn_value}。")
            shapes.append((geom, burn_val))
        
        if not shapes: 
            print(f"警告: Shapefile '{os.path.basename(shapefile_path)}' 中没有找到有效的几何要素进行栅格化（可能在重投影或验证后为空）。")
            return np.zeros((tiff_height, tiff_width), dtype=np.uint8), src.meta

        # 4. 一次性 rasterize 整张图的要素
        mask_full = rasterize(
            shapes=shapes,
            out_shape=(tiff_height, tiff_width),
            transform=tiff_transform,
            fill=0,
            dtype=np.uint8,
            all_touched=True
        )
        return mask_full, src.meta

def save_chips_from_arrays(tiff_path, mask_full, output_image_dir, output_mask_dir,
                           tile_size=512, image_bands_to_use=(1,2,3), nodata_fill_value=0,
                           background_threshold=0.999, min_valid_pixels=10, skip_background=True):
    """
    将 TIFF 按 tile_size 切片并保存，同时对齐 mask_full 的小窗口，
    并根据背景比例跳过大部分是背景的切片。
    """
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    total_chips = 0
    saved_chips = 0
    skipped_chips = 0

    with rasterio.open(tiff_path) as src:
        height, width = src.height, src.width

        for r in range(0, height, tile_size):
            for c in range(0, width, tile_size):
                total_chips += 1
                win_w = tile_size if c + tile_size <= width else (width - c)
                win_h = tile_size if r + tile_size <= height else (height - r)
                
                if win_w == 0 or win_h == 0:
                    continue

                window = Window(c, r, win_w, win_h)

                img_data = src.read(image_bands_to_use, window=window)
                nodata = src.nodata
                if nodata is not None:
                    for b_idx in range(img_data.shape[0]):
                        img_data[b_idx][img_data[b_idx] == nodata] = nodata_fill_value

                img_arr = np.moveaxis(img_data, 0, -1)
                
                if img_arr.dtype != np.uint8:
                    if np.issubdtype(img_arr.dtype, np.floating):
                        if img_arr.size > 0 and img_arr.min() >= 0 and img_arr.max() <= 1.0 and (img_arr.max() - img_arr.min()) > 1e-6 :
                            img_arr = (img_arr * 255).astype(np.uint8)
                        elif img_arr.size > 0 : # Check size before percentile
                            vmin, vmax = np.percentile(img_arr, [2, 98])
                            if vmax > vmin:
                                img_arr = np.clip(img_arr, vmin, vmax)
                                img_arr = ((img_arr - vmin) / (vmax - vmin) * 255).astype(np.uint8)
                            else:
                                img_arr = np.full_like(img_arr, 0, dtype=np.uint8)
                        else: # Empty array after windowing, or constant
                             img_arr = np.full_like(img_arr, 0, dtype=np.uint8)
                    elif np.issubdtype(img_arr.dtype, np.integer):
                        if img_arr.size > 0: # Check size before percentile
                            vmin, vmax = np.percentile(img_arr, [2, 98])
                            if vmax > vmin:
                                img_arr = np.clip(img_arr, vmin, vmax)
                                img_arr = ((img_arr - vmin) / (vmax - vmin) * 255).astype(np.uint8)
                            else:
                                img_arr = np.full_like(img_arr, 0, dtype=np.uint8)
                        else: # Empty array
                            img_arr = np.full_like(img_arr, 0, dtype=np.uint8)
                    else:
                        img_arr = img_arr.astype(np.uint8)

                if img_arr.ndim == 2 or (img_arr.ndim == 3 and img_arr.shape[2] == 1):
                    img_mode = 'L'
                    if img_arr.ndim == 3:
                        img_arr = img_arr.squeeze(axis=2)
                elif img_arr.ndim == 3 and img_arr.shape[2] == 3:
                    img_mode = 'RGB'
                elif img_arr.ndim == 3 and img_arr.shape[2] == 4:
                    img_mode = 'RGBA'
                else:
                    print(f"警告: 图像切片 {r}_{c} 的尺寸或波段数 ({img_arr.shape}) 不支持，已跳过。")
                    continue

                mask_chip_raw = mask_full[r:r+win_h, c:c+win_w]
                
                img_arr_to_save = img_arr # Assume img_arr is already padded if needed or handled by PIL
                mask_chip_padded = np.zeros((tile_size, tile_size), dtype=np.uint8)

                if img_arr.shape[0] != tile_size or img_arr.shape[1] != tile_size:
                    pad_h_img = tile_size - img_arr.shape[0]
                    pad_w_img = tile_size - img_arr.shape[1]
                    if img_mode == 'L':
                        img_arr_to_save = np.pad(img_arr, ((0, pad_h_img), (0, pad_w_img)), mode='constant', constant_values=0)
                    else:
                         img_arr_to_save = np.pad(img_arr, ((0, pad_h_img), (0, pad_w_img), (0,0)), mode='constant', constant_values=0)
                
                mask_chip_padded[:win_h, :win_w] = mask_chip_raw
                mask_to_save = (mask_chip_padded * 255).astype(np.uint8) 

                if skip_background and is_background_only(mask_chip_padded, background_threshold, min_valid_pixels):
                    skipped_chips += 1
                    continue

                tiff_base_name_for_chip = os.path.splitext(os.path.basename(tiff_path))[0]
                base_name = f"{r}_{c}"
                img_out_path = os.path.join(output_image_dir, base_name + ".png")
                mask_out_path = os.path.join(output_mask_dir, base_name + "_mask.png")

                Image.fromarray(img_arr_to_save, mode=img_mode).save(img_out_path)
                Image.fromarray(mask_to_save, mode='L').save(mask_out_path)
                saved_chips += 1
        
        print(f"  文件 '{os.path.basename(tiff_path)}' 切片统计:")
        print(f"    总切片数: {total_chips}")
        print(f"    保存切片数: {saved_chips}")
        print(f"    跳过背景切片数: {skipped_chips}")
        if total_chips > 0:
            print(f"    跳过比例: {skipped_chips/total_chips:.2%}")
            print(f"    有效切片比例: {saved_chips/total_chips:.2%}")
        else:
            print("    未生成任何切片。")


def main():
    tiff_input_directory = r"F:\project\unetpp_find\data\tiff\select"
    shapefile_global_path = r"F:\project\unetpp_find\data\GLID_annotation\annotation\annotation.shp" 
    output_root_directory = r"F:\project\unetpp_find\data\processed_chips" 

    config_tile_size = 512
    config_image_bands = (1, 2, 3)
    config_nodata_fill = 0
    config_bg_threshold = 0.999
    config_min_pixels = 10
    config_skip_bg = True

    search_pattern_tif = os.path.join(tiff_input_directory, "*.tif")
    search_pattern_tiff = os.path.join(tiff_input_directory, "*.tiff")
    tiff_file_list = glob.glob(search_pattern_tif) + glob.glob(search_pattern_tiff)

    if not tiff_file_list:
        print(f"在目录 '{tiff_input_directory}' 中没有找到任何TIFF文件。")
        return

    print(f"找到 {len(tiff_file_list)} 个TIFF文件，将开始处理...")

    for tiff_file in tiff_file_list:
        tiff_file_basename = os.path.splitext(os.path.basename(tiff_file))[0]
        print(f"\n--- 开始处理TIFF文件: {tiff_file_basename} ---")

        current_images_output_dir = os.path.join(output_root_directory, "images", tiff_file_basename)
        current_masks_output_dir = os.path.join(output_root_directory, "masks", tiff_file_basename)
        
        try:
            full_mask_array, _ = rasterize_full_mask(
                tiff_path=tiff_file,
                shapefile_path=shapefile_global_path,
                mask_burn_attribute=None,
                default_burn_value=1
            )
            print(f"  为 '{tiff_file_basename}' 生成的完整掩膜尺寸: {full_mask_array.shape}")
            unique_mask_values, counts = np.unique(full_mask_array, return_counts=True)
            print(f"  完整掩膜中的唯一值和计数: {dict(zip(unique_mask_values, counts))}")

            if np.sum(full_mask_array > 0) == 0:
                 print(f"  警告: 为 '{tiff_file_basename}' 生成的完整掩膜不包含任何前景目标。可能Shapefile与此TIFF不重叠，或Shapefile为空/无效，或重投影后无重叠。")
                 print(f"  将跳过对 '{tiff_file_basename}' 的切片保存。")
                 continue

            save_chips_from_arrays(
                tiff_path=tiff_file,
                mask_full=full_mask_array,
                output_image_dir=current_images_output_dir,
                output_mask_dir=current_masks_output_dir,
                tile_size=config_tile_size,
                image_bands_to_use=config_image_bands,
                nodata_fill_value=config_nodata_fill,
                background_threshold=config_bg_threshold,
                min_valid_pixels=config_min_pixels,
                skip_background=config_skip_bg
            )
            print(f"  '{tiff_file_basename}' 的切片已处理完毕。")

        except ValueError as ve: # Catch custom ValueErrors from CRS issues etc.
            print(f"  处理文件 '{tiff_file_basename}' 时发生配置或数据错误: {ve}")
            print(f"  跳过此文件。")
        except Exception as e:
            print(f"  处理文件 '{tiff_file_basename}' 时发生未预料的错误: {e}")
            traceback.print_exc()
            print(f"  跳过此文件。")
            
    print("\n所有TIFF文件处理完成。")

if __name__ == "__main__":
    main()