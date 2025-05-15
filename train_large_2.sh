# 定义配置文件和数据文件的数组
configs=(
  projects/yolo_dual/FLIR/2025_05_15/yolov8_mambatransformer.py
  projects/yolo_dual/FLIR/2025_05_15/yolov8_mambatransformer_enhance.py
)

# 遍历数组并运行每个 Python 脚本
for i in "${!configs[@]}"; do
  config=${configs[$i]}

  # 提取配置文件名，不包含路径和扩展名
  config_name=$(basename "$config" .py)

  dataset=$(echo "$config" | cut -d'/' -f3)

  # 提取日期部分 (第5个部分)
  date_part=$(echo "$config" | cut -d'/' -f4)

  # 生成目标工作目录路径
  work_dir="work_dirs/$dataset/$date_part/$config_name"

  # 最大重试次数
  max_retries=10
  retry_count=0

  while [ $retry_count -lt $max_retries ]; do
    echo "Running train script with config: $config..."

    # 执行训练脚本
    CUDA_VISIBLE_DEVICES=1 python tools/train.py "$config" --work-dir "$work_dir" --resume
    
    # 检查退出状态
    if [ $? -eq 0 ]; then
      echo "Training script with config $config finished successfully."
      break
    else
      retry_count=$((retry_count + 1))
      echo "Training script with config $config failed. Attempt $retry_count of $max_retries."
      if [ $retry_count -lt $max_retries ]; then
        echo "Waiting for 10 minutes before retrying..."
        sleep 600  # 休眠10分钟
      else
        echo "Max retries reached for config $config. Exiting."
        break
      fi
    fi
  done
done

echo "All training scripts finished."