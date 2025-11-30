from src.DriverNet.utils.depth_utils import compute_depth_groups, masked_avg_pool, masked_max_pool
from src.DriverNet.utils.download_dataset import download_kaggle_competition
from src.DriverNet.utils.ema import EMA
from src.DriverNet.utils.ensemble import average_predictions
from src.DriverNet.utils.infer_feature_dim import infer_feat_dim
from src.DriverNet.utils.logger import wandb_logger
from src.DriverNet.utils.losses import kl_div_with_temperature, get_hard_example_weights, entropy_gated_kd
from src.DriverNet.utils.submission import create_submission
from src.DriverNet.utils.visualization import save_confusion_matrix
# from src.DriverNet.utils.knn import (
#     build_knn_and_search,
#     compute_and_ensemble_knn_probabilities,
#     extract_features
# )

# from src.DriverNet.utils.segment_average import (
#     segment_average_test_images,
#     build_segment_graph,
#     apply_segment_average_to_probs,
#     apply_segment_average_to_predictions,
#     apply_segment_average_from_checkpoint,
#     apply_sequential_post_processing,
#     apply_sequential_post_processing_from_checkpoint,
#     generate_sequential_post_processing_submissions
# )
