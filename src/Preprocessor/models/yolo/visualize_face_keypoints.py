import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Tuple
import os


def analyze_class_keypoints_distribution(
    csv_path: str,
    save_path: Optional[str] = None,
    max_keypoints: int = 5
) -> Tuple[Dict, Dict]:
    df = pd.read_csv(csv_path)

    class_stats = {}
    class_normalized_keypoints = {}

    for classname in sorted(df['classname'].unique()):
        class_df = df[df['classname'] == classname]

        normalized_keypoints_list = []

        for _, row in class_df.iterrows():
            if pd.isna(row['bbox_x1']) or pd.isna(row['keypoint_0_x']):
                continue

            bbox_x1 = row['bbox_x1']
            bbox_y1 = row['bbox_y1']
            bbox_x2 = row['bbox_x2']
            bbox_y2 = row['bbox_y2']

            bbox_width = bbox_x2 - bbox_x1
            bbox_height = bbox_y2 - bbox_y1

            if bbox_width <= 0 or bbox_height <= 0:
                continue

            keypoints_normalized = []
            for i in range(max_keypoints):
                kp_x_col = f'keypoint_{i}_x'
                kp_y_col = f'keypoint_{i}_y'

                if pd.isna(row[kp_x_col]) or pd.isna(row[kp_y_col]):
                    break

                kp_x = row[kp_x_col]
                kp_y = row[kp_y_col]

                normalized_x = (kp_x - bbox_x1) / bbox_width
                normalized_y = (kp_y - bbox_y1) / bbox_height

                keypoints_normalized.append([normalized_x, normalized_y])

            if len(keypoints_normalized) > 0:
                normalized_keypoints_list.append(np.array(keypoints_normalized))

        if len(normalized_keypoints_list) > 0:
            all_keypoints = np.array(normalized_keypoints_list)
            class_normalized_keypoints[classname] = all_keypoints

            stats = {}
            for i in range(max_keypoints):
                if i < all_keypoints.shape[1]:
                    kp_data = all_keypoints[:, i, :]
                    stats[f'keypoint_{i}'] = {
                        'mean_x': np.mean(kp_data[:, 0]),
                        'mean_y': np.mean(kp_data[:, 1]),
                        'std_x': np.std(kp_data[:, 0]),
                        'std_y': np.std(kp_data[:, 1]),
                        'count': len(kp_data)
                    }

            class_stats[classname] = stats

    num_classes = len(class_stats)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    keypoint_names = ['Left Eye', 'Right Eye', 'Nose', 'Left Mouth', 'Right Mouth']
    colors = ['blue', 'green', 'red', 'orange', 'purple']

    for idx, (classname, stats) in enumerate(sorted(class_stats.items())):
        ax = axes[idx]

        if classname in class_normalized_keypoints:
            all_keypoints = class_normalized_keypoints[classname]

            for i, (kp_name, color) in enumerate(zip(keypoint_names, colors)):
                if i < all_keypoints.shape[1]:
                    kp_data = all_keypoints[:, i, :]

                    ax.scatter(kp_data[:, 0], kp_data[:, 1],
                             alpha=0.3, s=10, color=color, label=kp_name)

                    mean_x = stats[f'keypoint_{i}']['mean_x']
                    mean_y = stats[f'keypoint_{i}']['mean_y']
                    ax.scatter(mean_x, mean_y, s=200, marker='x',
                             color='black', linewidths=3, zorder=10)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_title(f'{classname}\n(n={stats["keypoint_0"]["count"]})',
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Normalized X')
        ax.set_ylabel('Normalized Y')
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()

        if idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    plt.tight_layout()

    if save_path:
        scatter_path = save_path.replace('.png', '_scatter.png') if save_path.endswith('.png') else save_path + '_scatter.png'
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        print(f"산점도 시각화 저장: {scatter_path}")
    else:
        plt.show()

    plt.close()

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for idx, (classname, stats) in enumerate(sorted(class_stats.items())):
        ax = axes[idx]

        if classname in class_normalized_keypoints:
            all_keypoints = class_normalized_keypoints[classname]

            x_coords = []
            y_coords = []
            for i in range(min(max_keypoints, all_keypoints.shape[1])):
                kp_data = all_keypoints[:, i, :]
                x_coords.extend(kp_data[:, 0])
                y_coords.extend(kp_data[:, 1])

            if len(x_coords) > 0:
                ax.hexbin(x_coords, y_coords, gridsize=20, cmap='YlOrRd', mincnt=1)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_aspect('equal')

        ax.set_title(f'{classname} - Heatmap', fontsize=12, fontweight='bold')
        ax.set_xlabel('Normalized X')
        ax.set_ylabel('Normalized Y')
        ax.invert_yaxis()

    plt.tight_layout()

    if save_path:
        heatmap_path = save_path.replace('.png', '_heatmap.png') if save_path.endswith('.png') else save_path + '_heatmap.png'
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        print(f"히트맵 시각화 저장: {heatmap_path}")
    else:
        plt.show()

    plt.close()

    print("\n클래스별 Keypoints 통계:")
    for classname, stats in sorted(class_stats.items()):
        print(f"\n{classname}:")
        for i, kp_name in enumerate(keypoint_names):
            if f'keypoint_{i}' in stats:
                s = stats[f'keypoint_{i}']
                print(f"  {kp_name}: mean=({s['mean_x']:.3f}, {s['mean_y']:.3f}), "
                      f"std=({s['std_x']:.3f}, {s['std_y']:.3f}), count={s['count']}")

    return class_stats, class_normalized_keypoints


def compare_keypoints_across_classes(
    csv_path: str,
    save_path: Optional[str] = None,
    max_keypoints: int = 5
):
    df = pd.read_csv(csv_path)

    keypoint_names = ['Left Eye', 'Right Eye', 'Nose', 'Left Mouth', 'Right Mouth']

    fig, axes = plt.subplots(1, max_keypoints, figsize=(4 * max_keypoints, 6))
    if max_keypoints == 1:
        axes = [axes]

    for kp_idx in range(max_keypoints):
        ax = axes[kp_idx]

        class_means_x = []
        class_means_y = []
        class_names = []

        for classname in sorted(df['classname'].unique()):
            class_df = df[df['classname'] == classname]

            normalized_x_list = []
            normalized_y_list = []

            for _, row in class_df.iterrows():
                if pd.isna(row['bbox_x1']) or pd.isna(row[f'keypoint_{kp_idx}_x']):
                    continue

                bbox_x1 = row['bbox_x1']
                bbox_y1 = row['bbox_y1']
                bbox_x2 = row['bbox_x2']
                bbox_y2 = row['bbox_y2']

                bbox_width = bbox_x2 - bbox_x1
                bbox_height = bbox_y2 - bbox_y1

                if bbox_width <= 0 or bbox_height <= 0:
                    continue

                kp_x = row[f'keypoint_{kp_idx}_x']
                kp_y = row[f'keypoint_{kp_idx}_y']

                normalized_x = (kp_x - bbox_x1) / bbox_width
                normalized_y = (kp_y - bbox_y1) / bbox_height

                normalized_x_list.append(normalized_x)
                normalized_y_list.append(normalized_y)

            if len(normalized_x_list) > 0:
                class_means_x.append(np.mean(normalized_x_list))
                class_means_y.append(np.mean(normalized_y_list))
                class_names.append(classname)

        if len(class_means_x) > 0:
            ax.scatter(class_means_x, class_means_y, s=100, alpha=0.7)

            for i, classname in enumerate(class_names):
                ax.annotate(classname, (class_means_x[i], class_means_y[i]),
                          fontsize=8, ha='center', va='bottom')

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_title(f'{keypoint_names[kp_idx]}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Normalized X')
        ax.set_ylabel('Normalized Y')
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()

    plt.tight_layout()

    if save_path:
        compare_path = save_path.replace('.png', '_compare.png') if save_path.endswith('.png') else save_path + '_compare.png'
        plt.savefig(compare_path, dpi=300, bbox_inches='tight')
        print(f"비교 시각화 저장: {compare_path}")
    else:
        plt.show()

    plt.close()


def calculate_yaw_pitch_roll(
    left_eye: np.ndarray,
    right_eye: np.ndarray,
    nose: np.ndarray,
    left_mouth: Optional[np.ndarray] = None,
    right_mouth: Optional[np.ndarray] = None
) -> Tuple[float, float, float]:
    eye_center = (left_eye + right_eye) / 2.0
    eye_vector = right_eye - left_eye
    eye_distance = np.linalg.norm(eye_vector)

    if eye_distance == 0:
        return 0.0, 0.0, 0.0

    roll = np.arctan2(eye_vector[1], eye_vector[0]) * 180.0 / np.pi

    nose_to_eye_center = nose - eye_center
    vertical_distance = np.linalg.norm(nose_to_eye_center)

    if vertical_distance == 0:
        pitch = 0.0
    else:
        pitch_rad = np.arcsin(nose_to_eye_center[1] / vertical_distance)
        pitch = pitch_rad * 180.0 / np.pi

    left_eye_to_nose = np.linalg.norm(nose - left_eye)
    right_eye_to_nose = np.linalg.norm(nose - right_eye)

    if left_eye_to_nose + right_eye_to_nose == 0:
        yaw = 0.0
    else:
        yaw_ratio = (right_eye_to_nose - left_eye_to_nose) / (left_eye_to_nose + right_eye_to_nose)
        yaw = np.arcsin(np.clip(yaw_ratio, -1, 1)) * 180.0 / np.pi

    return yaw, pitch, roll


def analyze_head_pose_distribution(
    csv_path: str,
    save_path: Optional[str] = None,
    max_keypoints: int = 5
) -> Tuple[Dict, Dict]:
    df = pd.read_csv(csv_path)

    class_poses = {}
    class_stats = {}

    for classname in sorted(df['classname'].unique()):
        class_df = df[df['classname'] == classname]

        yaws = []
        pitches = []
        rolls = []

        for _, row in class_df.iterrows():
            if pd.isna(row['bbox_x1']) or pd.isna(row['keypoint_0_x']):
                continue

            try:
                left_eye = np.array([row['keypoint_0_x'], row['keypoint_0_y']])
                right_eye = np.array([row['keypoint_1_x'], row['keypoint_1_y']])
                nose = np.array([row['keypoint_2_x'], row['keypoint_2_y']])

                if pd.isna(left_eye[0]) or pd.isna(right_eye[0]) or pd.isna(nose[0]):
                    continue

                left_mouth = None
                right_mouth = None
                if max_keypoints >= 5:
                    if not pd.isna(row['keypoint_3_x']) and not pd.isna(row['keypoint_4_x']):
                        left_mouth = np.array([row['keypoint_3_x'], row['keypoint_3_y']])
                        right_mouth = np.array([row['keypoint_4_x'], row['keypoint_4_y']])

                yaw, pitch, roll = calculate_yaw_pitch_roll(
                    left_eye, right_eye, nose, left_mouth, right_mouth
                )

                yaws.append(yaw)
                pitches.append(pitch)

            except Exception as e:
                continue

        if len(yaws) > 0:
            class_poses[classname] = {
                'yaw': np.array(yaws),
                'pitch': np.array(pitches),
                'roll': np.array(rolls)
            }

            class_stats[classname] = {
                'yaw_mean': np.mean(yaws), 'yaw_std': np.std(yaws),
                'pitch_mean': np.mean(pitches), 'pitch_std': np.std(pitches),
                'roll_mean': np.mean(rolls), 'roll_std': np.std(rolls),
                'count': len(yaws)
            }

    if len(class_poses) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for i, metric in enumerate(['yaw', 'pitch', 'roll']):
            ax = axes[i]
            data = []
            labels = []

            for classname in sorted(class_poses.keys()):
                data.append(class_poses[classname][metric])
                labels.append(classname)

            ax.boxplot(data, labels=labels)
            ax.set_title(f'{metric.capitalize()} Distribution')
            ax.set_xlabel('Class')
            ax.set_ylabel('Angle (degrees)')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            pose_path = save_path.replace('.png', '_pose.png') if save_path.endswith('.png') else save_path + '_pose.png'
            plt.savefig(pose_path, dpi=300, bbox_inches='tight')
            print(f"자세 분포 시각화 저장: {pose_path}")
        else:
            plt.show()

        plt.close()

    return class_stats, class_poses


if __name__ == "__main__":
    csv_path = "data/train_face_keypoints.csv"

    if not os.path.exists(csv_path):
        print(f"CSV 파일을 찾을 수 없습니다: {csv_path}")
        print("먼저 extract_face_keypoints.py를 실행하여 keypoints를 추출하세요.")
    else:
        save_path = "data/class_keypoints_distribution"
        analyze_class_keypoints_distribution(csv_path, save_path=save_path)
        compare_keypoints_across_classes(csv_path, save_path=save_path)
