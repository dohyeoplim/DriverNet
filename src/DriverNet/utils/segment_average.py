import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Tuple, List, Optional, Dict, Any
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

from src.DriverNet.utils.knn import (
    build_knn_and_search,
    compute_and_ensemble_knn_probabilities,
    extract_features
)
from src.DriverNet.utils.inference_utils import extract_test_features_and_probs
from src.DriverNet.models.base import BaseModel
from src.DriverNet.utils.submission import create_submission
from src.DriverNet.utils.tta import predict_with_tta

def segment_average_test_images(
    test_features: np.ndarray,
    test_probs: np.ndarray,
    distance_threshold: float = 0.03,
    metric: str = 'cosine',
    min_group_size: int = 2,
    confidence_threshold: float = 0.9,
    max_candidates: int = 200
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    print("\n" + "=" * 60)
    print("세그먼트 평균: 유사한 테스트 이미지 그룹화")
    print("=" * 60)

    print(f"입력 형태: {test_features.shape}")
    print(f"거리 임계값: {distance_threshold}")
    print(f"거리 메트릭: {metric}")
    print(f"최소 그룹 크기: {min_group_size}")
    print(f"확신도 임계값: {confidence_threshold}")

    print(f"\n유사도 그래프 구축 중 (임계값={distance_threshold})...")

    from sklearn.neighbors import radius_neighbors_graph

    adjacency_matrix = radius_neighbors_graph(
        test_features,
        radius=distance_threshold,
        metric=metric,
        mode='connectivity',
        include_self=True,
        n_jobs=-1
    )

    print(f"연결된 컴포넌트 찾는 중...")
    n_components, segment_labels = connected_components(
        csgraph=adjacency_matrix,
        directed=False,
        return_labels=True
    )

    print(f"\n{len(test_features)}개의 이미지에서 {n_components}개의 세그먼트 발견.")

    num_classes = test_probs.shape[1]
    prob_cols = list(range(num_classes))

    df = pd.DataFrame(test_probs, columns=prob_cols)
    df['segment_id'] = segment_labels

    segment_counts = df.groupby('segment_id')['segment_id'].transform('count')

    segment_means = df.groupby('segment_id')[prob_cols].transform('mean')

    segment_max_conf = segment_means[prob_cols].max(axis=1)

    mask = (
        (segment_counts >= min_group_size) &
        (segment_max_conf >= confidence_threshold)
    ).values.reshape(-1, 1)

    final_probs = np.where(mask, segment_means.values, test_probs)

    final_probs = np.clip(final_probs, 1e-8, 1.0 - 1e-8)
    final_probs = final_probs / final_probs.sum(axis=1, keepdims=True)

    segment_sizes = df.groupby('segment_id').size()
    n_singletons = (segment_sizes == 1).sum()
    n_large_segments = (segment_sizes >= 10).sum()
    max_segment_size = segment_sizes.max()

    n_confident_groups = ((segment_counts >= min_group_size) & (segment_max_conf >= confidence_threshold)).sum()

    stats = {
        'n_components': n_components,
        'segment_sizes': segment_sizes.values,
        'n_singletons': n_singletons,
        'n_large_segments': n_large_segments,
        'max_segment_size': max_segment_size,
        'n_averaged': n_confident_groups,
        'n_confident_groups': n_confident_groups,
        'confidence_threshold': confidence_threshold
    }

    return final_probs, segment_labels, stats


def build_segment_graph(
    test_features: np.ndarray,
    distance_threshold: float = 0.03,
    metric: str = 'cosine',
    max_candidates: int = 200
) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    """
    Radius Neighbors Graph
    """
    from sklearn.neighbors import radius_neighbors_graph

    adjacency_matrix = radius_neighbors_graph(
        test_features,
        radius=distance_threshold,
        metric=metric,
        mode='connectivity',
        include_self=True,
        n_jobs=-1
    )

    n_components, segment_labels = connected_components(
        csgraph=adjacency_matrix,
        directed=False,
        return_labels=True
    )

    segment_sizes = pd.Series(segment_labels).value_counts().sort_index()
    n_singletons = (segment_sizes == 1).sum()
    n_large_segments = (segment_sizes >= 10).sum()
    max_segment_size = segment_sizes.max()

    stats = {
        'n_components': n_components,
        'segment_sizes': segment_sizes.values,
        'n_singletons': n_singletons,
        'n_large_segments': n_large_segments,
        'max_segment_size': max_segment_size
    }

    return segment_labels, n_components, stats


def apply_segment_average_to_probs(
    test_probs: np.ndarray,
    segment_labels: np.ndarray,
    min_group_size: int = 2,
    confidence_threshold: float = 0.9
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    이미 구축된 segment_labels를 사용하여 확률을 평균화합니다。
    """
    num_classes = test_probs.shape[1]
    prob_cols = list(range(num_classes))

    df = pd.DataFrame(test_probs, columns=prob_cols)
    df['segment_id'] = segment_labels

    segment_counts = df.groupby('segment_id')['segment_id'].transform('count')
    segment_means = df.groupby('segment_id')[prob_cols].transform('mean')
    segment_max_conf = segment_means[prob_cols].max(axis=1)

    mask = (
        (segment_counts >= min_group_size) &
        (segment_max_conf >= confidence_threshold)
    ).values.reshape(-1, 1)

    final_probs = np.where(
        mask,
        segment_means.values,
        test_probs
    )

    final_probs = np.clip(final_probs, 1e-8, 1.0 - 1e-8)
    final_probs = final_probs / final_probs.sum(axis=1, keepdims=True)

    n_confident_groups = ((segment_counts >= min_group_size) & (segment_max_conf >= confidence_threshold)).sum()
    stats = {
        'n_averaged': n_confident_groups,
        'n_confident_groups': n_confident_groups,
        'confidence_threshold': confidence_threshold
    }

    return final_probs, stats


def apply_segment_average_to_predictions(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    distance_threshold: float = 0.03,
    metric: str = 'cosine',
    min_group_size: int = 2,
    confidence_threshold: float = 0.9,
    ms_feature_dim: Optional[int] = None,
    temperature: float = 1.0
) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    """
    모델과 데이터로더를 받아 Segment Average를 적용합니다。
    """
    print("\n" + "=" * 60)
    print("예측에 세그먼트 평균 적용")
    print("=" * 60)

    test_features, test_probs, test_image_names = extract_test_features_and_probs(
        model=model,
        test_loader=test_loader,
        device=device,
        ms_feature_dim=ms_feature_dim,
        temperature=temperature
    )

    if isinstance(test_features, torch.Tensor):
        test_features_np = test_features.cpu().numpy()
    else:
        test_features_np = np.asarray(test_features)

    if isinstance(test_probs, torch.Tensor):
        test_probs_np = test_probs.cpu().numpy()
    else:
        test_probs_np = np.asarray(test_probs)

    test_features_np = normalize(test_features_np, norm='l2', axis=1)

    averaged_probs, segment_labels, stats = segment_average_test_images(
        test_features=test_features_np,
        test_probs=test_probs_np,
        distance_threshold=distance_threshold,
        metric=metric,
        min_group_size=min_group_size,
        confidence_threshold=confidence_threshold
    )

    stats['image_names'] = test_image_names
    stats['segment_labels'] = segment_labels

    return averaged_probs, test_image_names, stats


def apply_segment_average_from_checkpoint(
    model_path: str,
    test_loader: DataLoader,
    device: torch.device,
    distance_threshold: float = 0.03,
    metric: str = 'cosine',
    min_group_size: int = 2,
    confidence_threshold: float = 0.9,
) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    print("\n" + "=" * 60)
    print("체크포인트에서 세그먼트 평균 적용")
    print("=" * 60)
    print(f"모델 경로: {model_path}")

    model = BaseModel.load_from_checkpoint(
        checkpoint_path=model_path,
        map_location=device
    )
    model = model.to(device)
    model.eval()

    averaged_probs, image_names, stats = apply_segment_average_to_predictions(
        model=model,
        test_loader=test_loader,
        device=device,
        distance_threshold=distance_threshold,
        metric=metric,
        min_group_size=min_group_size,
        confidence_threshold=confidence_threshold,
        ms_feature_dim=model.hparams.get('ms_feature_dim'), # type: ignore
        temperature=model.hparams.get('temperature', 1.0) # type: ignore
    )

    return averaged_probs, image_names, stats


def apply_sequential_post_processing(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    knn_k: int = 50,
    knn_alpha: float = 0.3,
    knn_temperature: float = 0.1,
    seg_threshold: float = 0.03,
    seg_metric: str = 'cosine',
    seg_min_size: int = 2,
    seg_confidence_threshold: float = 0.9,
    ms_feature_dim: Optional[int] = None,
    temperature: float = 1.0
) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    """
    Model -> KNN -> Segment Average 순차 적용 파이프라인
    """
    print("\n" + "=" * 70)
    print(" 순차적 후처리 파이프라인 (Model -> KNN -> Segment Average)")
    print("=" * 70)

    print("\n 특징 및 확률 추출 중...")
    train_features_data = extract_features(
        model=model,
        dataloader=train_loader,
        device=device
    )
    val_features_data = extract_features(
        model=model,
        dataloader=val_loader,
        device=device
    )
    all_features = np.concatenate([train_features_data["features"], val_features_data["features"]])
    all_labels = np.concatenate([train_features_data["labels"], val_features_data["labels"]])
    print("   - Test 세트에서 추출 중...")
    test_features, test_probs_raw, test_image_names = extract_test_features_and_probs(
        model=model,
        test_loader=test_loader,
        device=device,
        ms_feature_dim=ms_feature_dim,
        temperature=temperature
    )

    if isinstance(test_features, torch.Tensor):
        test_features_np = test_features.cpu().numpy()
    else:
        test_features_np = np.asarray(test_features)

    if isinstance(test_probs_raw, torch.Tensor):
        test_probs_raw_np = test_probs_raw.cpu().numpy()
    else:
        test_probs_raw_np = np.asarray(test_probs_raw)

    print(f"   - Train+Val 특징: {all_features.shape}, 레이블: {all_labels.shape}")
    print(f"   - Test 특징: {test_features_np.shape}, 확률: {test_probs_raw_np.shape}")

    print("\nL2 정규화 적용 중...")
    all_features = normalize(all_features, norm='l2', axis=1)
    test_features_np = normalize(test_features_np, norm='l2', axis=1)

    print(f"\n KNN 보정 적용 중 (k={knn_k}, alpha={knn_alpha})...")
    max_distances, max_indices = build_knn_and_search(
        all_features=all_features,
        test_features=test_features_np,
        max_k=knn_k
    )

    distances = max_distances[:, :knn_k]
    indices = max_indices[:, :knn_k]

    test_probs_knn = compute_and_ensemble_knn_probabilities(
        distances=distances,
        indices=indices,
        all_labels=all_labels,
        test_probs=test_probs_raw_np,
        k=knn_k,
        temperature=knn_temperature,
        alpha=knn_alpha
    )

    print(f"   - KNN 보정 완료")

    del max_distances, max_indices, distances, indices
    import gc
    gc.collect()

    print(f"\n3️⃣ 세그먼트 평균 적용 중 (임계값={seg_threshold}, 최소 크기={seg_min_size}, 확신도={seg_confidence_threshold})...")
    final_probs, segment_labels, stats = segment_average_test_images(
        test_features=test_features_np,
        test_probs=test_probs_knn,
        distance_threshold=seg_threshold,
        metric=seg_metric,
        min_group_size=seg_min_size,
        confidence_threshold=seg_confidence_threshold
    )

    print(f"\n{'='*70}")
    print("순차적 후처리 완료!")
    print(f"{ '='*70}")

    stats['image_names'] = test_image_names
    stats['segment_labels'] = segment_labels
    stats['knn_k'] = knn_k
    stats['knn_alpha'] = knn_alpha

    return final_probs, test_image_names, stats


def apply_sequential_post_processing_from_checkpoint(
    model_path: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    knn_k: int = 50,
    knn_alpha: float = 0.3,
    knn_temperature: float = 0.1,
    seg_threshold: float = 0.03,
    seg_metric: str = 'cosine',
    seg_min_size: int = 2,
    seg_confidence_threshold: float = 0.9,
) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    print("\n" + "=" * 70)
    print("체크포인트에서 순차적 후처리 실행")
    print("=" * 70)
    print(f"모델 경로: {model_path}")

    model = BaseModel.load_from_checkpoint(
        checkpoint_path=model_path,
        map_location=device
    )
    model = model.to(device)
    model.eval()

    final_probs, image_names, stats = apply_sequential_post_processing(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        knn_k=knn_k,
        knn_alpha=knn_alpha,
        knn_temperature=knn_temperature,
        seg_threshold=seg_threshold,
        seg_metric=seg_metric,
        seg_min_size=seg_min_size,
        seg_confidence_threshold=seg_confidence_threshold,
        ms_feature_dim=model.hparams.get('ms_feature_dim'), # type: ignore
        temperature=model.hparams.get('temperature', 1.0) # type: ignore
    )

    return final_probs, image_names, stats


def generate_sequential_post_processing_submissions(
    model_path: str,
    config: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    knn_k_values: List[int],
    seg_threshold_values: List[float],
    knn_alpha: float = 0.3,
    knn_temperature: float = 0.1,
    seg_metric: str = 'cosine',
    seg_min_size: int = 2,
    seg_confidence_threshold: float = 0.9,
    ms_feature_dim: Optional[int] = None,
    temperature: float = 1.0,
    output_dir: Optional[str] = None,
    use_tta: bool = True
) -> None:
    from tqdm import tqdm
    from src.DriverNet.utils.submission import create_submission
    from src.DriverNet.utils.tta import predict_with_tta
    import os

    print("\n" + "=" * 70)
    print("순차적 후처리 제출 파일 생성")
    print("(다중 k 값, 다중 임계값)")
    if use_tta:
        print("TTA (Test Time Augmentation) 포함 - 특징: 원본, 확률: TTA")
    print("=" * 70)
    print(f"모델 경로: {model_path}")
    print(f"K 값 목록: {knn_k_values}")
    print(f"임계값 목록: {seg_threshold_values}")
    print(f"총 조합 수: {len(knn_k_values)} × {len(seg_threshold_values)} = {len(knn_k_values) * len(seg_threshold_values)}")

    if output_dir is None:
        output_dir = os.path.dirname(model_path)
    model_folder_name = os.path.basename(output_dir)
    model_file_basename = os.path.basename(model_path).replace('.pth', '').replace('.pt', '')

    model = load_model_from_checkpoint(
        model_file=model_path,
        config=config,
        device=device
    )

    print("\n 특징 및 확률 추출 중 (일회성)...")
    print("   - Train + Validation (KNN 참조 라이브러리)에서 추출 중...")
    train_features_data = extract_features(
        model=model,
        dataloader=train_loader,
        device=device
    )
    val_features_data = extract_features(
        model=model,
        dataloader=val_loader,
        device=device
    )
    all_features = np.concatenate([train_features_data["features"], val_features_data["features"]])
    all_labels = np.concatenate([train_features_data["labels"], val_features_data["labels"]])
    print("   - Test 세트에서 추출 중...")

    if use_tta:
        print("     [1단계] 원본 이미지에서 특징 추출 (TTA 없음)...")
        test_features, _, test_image_names = extract_test_features_and_probs(
            model=model,
            test_loader=test_loader,
            device=device,
            ms_feature_dim=ms_feature_dim,
            temperature=1.0
        )

        if isinstance(test_features, torch.Tensor):
            test_features_np = test_features.cpu().numpy()
        else:
            test_features_np = np.asarray(test_features)

        print("     [2단계] TTA로 확률 추출 (5 views)...")

        test_image_paths = []
        test_face_paths = []
        test_hand_paths = []
        test_depth_paths = []
        test_image_names = []

        if hasattr(test_loader.dataset, 'image_paths'):
            test_image_paths = test_loader.dataset.image_paths
            if hasattr(test_loader.dataset, 'face_paths'):
                test_face_paths = test_loader.dataset.face_paths
            if hasattr(test_loader.dataset, 'hand_paths'):
                test_hand_paths = test_loader.dataset.hand_paths
            if hasattr(test_loader.dataset, 'depth_paths'):
                test_depth_paths = test_loader.dataset.depth_paths
            if hasattr(test_loader.dataset, 'image_names'):
                test_image_names = test_loader.dataset.image_names

        test_probs_raw_np = predict_with_tta(
            model=model,
            test_image_paths=test_image_paths,
            test_face_paths=test_face_paths if test_face_paths else None,
            test_hand_paths=test_hand_paths if test_hand_paths else None,
            test_depth_paths=test_depth_paths if test_depth_paths else None,
            test_image_names=test_image_names,
            device=device,
            batch_size=test_loader.batch_size,
            num_workers=test_loader.num_workers
        )

    else:
        test_features, test_probs_raw, test_image_names = extract_test_features_and_probs(
            model=model,
            test_loader=test_loader,
            device=device,
            ms_feature_dim=ms_feature_dim,
            temperature=temperature
        )

        if isinstance(test_features, torch.Tensor):
            test_features_np = test_features.cpu().numpy()
        else:
            test_features_np = np.asarray(test_features)

        if isinstance(test_probs_raw, torch.Tensor):
            test_probs_raw_np = test_probs_raw.cpu().numpy()
        else:
            test_probs_raw_np = np.asarray(test_probs_raw)

    print(f"   - Train+Val 특징: {all_features.shape}, 레이블: {all_labels.shape}")
    print(f"   - Test 특징: {test_features_np.shape}, 확률: {test_probs_raw_np.shape}")

    print("\nL2 정규화 적용 중...")
    all_features = normalize(all_features, norm='l2', axis=1)
    test_features_np = normalize(test_features_np, norm='l2', axis=1)

    print(f"\n 그리드 탐색 시작 (K 값 x 임계값)...")

    max_k = max(knn_k_values)
    print(f"\n[최적화] Classification KNN 검색 1회 실행 (k={max_k})...")

    full_knn_dists, full_knn_indices = build_knn_and_search(
        all_features=all_features,
        test_features=test_features_np,
        max_k=max_k
    )

    max_candidates = 200
    print(f"[최적화] Segment Graph용 KNN 검색 1회 실행 (k={max_candidates})...")

    seg_nn = NearestNeighbors(n_neighbors=max_candidates, metric=seg_metric, n_jobs=-1)
    seg_nn.fit(test_features_np)
    seg_dists, seg_indices = seg_nn.kneighbors(test_features_np)

    for k in knn_k_values:
        print(f"\n[KNN] 처리 중 k={k}...")

        distances = full_knn_dists[:, :k]
        indices = full_knn_indices[:, :k]

        test_probs_knn = compute_and_ensemble_knn_probabilities(
            distances=distances,
            indices=indices,
            all_labels=all_labels,
            test_probs=test_probs_raw_np,
            k=k,
            temperature=knn_temperature,
            alpha=knn_alpha
        )

        for th in seg_threshold_values:
            print(f"  [SegAvg] 처리 중 threshold={th}...")

            mask = seg_dists <= th

            num_test = test_features_np.shape[0]
            row_indices = np.repeat(np.arange(num_test), mask.sum(axis=1))
            col_indices = seg_indices[mask]
            data = np.ones(len(col_indices), dtype=np.int8)

            adj_matrix_th = csr_matrix(
                (data, (row_indices, col_indices)),
                shape=(num_test, num_test)
            )

            adj_matrix_th = (adj_matrix_th + adj_matrix_th.T) > 0

            n_components, segment_labels = connected_components(
                csgraph=adj_matrix_th,
                directed=False,
                return_labels=True
            )

            final_probs, _ = apply_segment_average_to_probs(
                test_probs=test_probs_knn,
                segment_labels=segment_labels,
                min_group_size=seg_min_size,
                confidence_threshold=seg_confidence_threshold
            )

            output_filename = f"{model_folder_name}_{model_file_basename}_k{k}_th{th}.csv"
            output_path = os.path.join(output_dir, output_filename)

            create_submission(
                image_names=test_image_names,
                probabilities=final_probs,
                output_path=output_path
            )

    print(f"\n{'='*70}")
    print(f"모든 조합 완료! ({len(knn_k_values) * len(seg_threshold_values)} 파일)")
    print(f"저장 경로: {output_dir}")
    print(f"{ '='*70}")
