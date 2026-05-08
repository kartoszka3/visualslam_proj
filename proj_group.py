import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_camera_matrix_simple(video_path):
    cap = cv2.VideoCapture(video_path)
    _, frame = cap.read()
    cap.release()
    
    h, w = frame.shape[:2]
    focal_length = w * 1.2
    cx = w / 2
    cy = h / 2
    
    K = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    dist = np.array([-0.2, 0.05, 0, 0, 0], dtype=np.float32)
    
    return K, dist


def orb_descriptor(image):
    orb = cv2.ORB_create(
        nfeatures=1000,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=31,
        firstLevel=0,
        WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE,
        patchSize=31,
        fastThreshold=20
    )
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors


def match_features(des1, des2, kp1, kp2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    
    pts1 = []
    pts2 = []
    
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.75 * n.distance:
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(kp1[m.queryIdx].pt)
    
    return np.int32(pts1), np.int32(pts2)

def select_frames(video_path, frame_gap=10, max_frames=None):
    """
    Wczytaj wyselekcjonowane klatki z pliku video
    
    Args:
        video_path: ścieżka do pliku video
        frame_gap: pobierz co frame_gap-tą klatkę
        max_frames: maksymalna liczba klatek do wczytania (None = wszystkie)
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    selected_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Sprawdź limit klatek
        if max_frames is not None and selected_frames >= max_frames:
            break
        
        if frame_count % frame_gap == 0:
            frames.append(frame)
            selected_frames += 1
        
        frame_count += 1
    
    cap.release()
    
    return np.array(frames)

def o3d_visualization(points, rotations, translations):
    """
    Wizualizuj trajektorię + chmurę punktów 3D za pomocą open3d
    
    Args:
        points: macierz punktów 3D (N, 3)
        rotations: lista macierzy rotacji
        translations: lista wektorów translacji
    """
    try:
        import open3d as o3d
    except ImportError:
        print("Open3D nie jest zainstalowany. Zainstaluj: pip install open3d")
        return
    
    # Utwórz wizualizację
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="SLAM Reconstruction", width=1200, height=900)
    
    # 1. Dodaj chmurę punktów
    if points is not None and len(points) > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        # Koloruj punkty na zielono
        pcd.paint_uniform_color([0.0, 1.0, 0.0])
        vis.add_geometry(pcd)
        print(f"Dodano {len(points)} punktów 3D (zielone)")
    
    # 2. Dodaj trajektorię kamery
    if translations is not None and len(translations) > 0:
        # Przetwórz translacje na pozycje kamer
        camera_positions = []
        for t in translations:
            # t jest wektorem translacji (3,)
            camera_positions.append(t)
        
        camera_positions = np.array(camera_positions)
        
        # Utwórz punkty trajektorii
        traj_pcd = o3d.geometry.PointCloud()
        traj_pcd.points = o3d.utility.Vector3dVector(camera_positions)
        # Koloruj trajektorię na czerwono
        traj_pcd.paint_uniform_color([1.0, 0.0, 0.0])
        vis.add_geometry(traj_pcd)
        print(f"Dodano {len(camera_positions)} pozycji kamer (czerwone)")
        
        # Utwórz linie łączące pozycje kamer (trajektoria)
        if len(camera_positions) > 1:
            lines = []
            for i in range(len(camera_positions) - 1):
                lines.append([i, i + 1])
            
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(camera_positions)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.paint_uniform_color([1.0, 0.5, 0.0])  # Pomarańczowe linie
            vis.add_geometry(line_set)
            print(f"Dodano trajektorię (pomarańczowe linie)")
    
    # 3. Dodaj osie współrzędnych na początku
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    vis.add_geometry(mesh_frame)
    print("Dodano osie współrzędnych")
    
    # Ustawy opcje widoku
    print("\nWizualizacja gotowa!")
    print("Sterowanie:")
    print("  - Lewy przycisk myszy: obracaj widok")
    print("  - Prawy przycisk myszy: pan (przesuwaj)")
    print("  - Scroll: zoom")
    print("  - Naciśnij Q, aby zamknąć okno")
    
    vis.run()
    vis.destroy_window()


def pipeline(frames, K, dist):
    """
    Przetwórz konsekutywne pary klatek
    
    Args:
        frames: tablica wczytanych klatek
        K: macierz kamery
        dist: współczynniki dystorsji
    """
    print("Camera Matrix K:\n", K)
    print("Distortion Coefficients:\n", dist)
    
    print(f"Wczytano {len(frames)} klatek")
    print(f"Przetworzę {len(frames) - 1} par klatek")
    
    # Przechowuj wyniki dla każdej pary
    all_rotations = []
    all_translations = []
    all_points_3d = []
    
    # Przetwórz każdą parę konsekutywnych klatek
    for i in range(len(frames) - 1):
        print(f"\n--- Przetwarzam parę klatek {i} i {i+1} ---")
        
        img1 = frames[i]
        img2 = frames[i + 1]
        
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Oblicz deskryptory
        kp1, des1 = orb_descriptor(gray1)
        kp2, des2 = orb_descriptor(gray2)
        
        if des1 is None or des2 is None:
            print(f"Brak deskryptorów w parze {i}, {i+1}. Pomijam.")
            continue
        
        # Połącz cechy
        pts1, pts2 = match_features(des1, des2, kp1, kp2)
        
        if len(pts1) < 8:
            print(f"Za mało dopasowanych punktów ({len(pts1)}). Pomijam.")
            continue
        
        # Macierz fundamentalna
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
        if F is None:
            print(f"Nie można obliczyć macierzy fundamentalnej. Pomijam.")
            continue
        
        pts1 = pts1[mask.ravel() == 1]
        pts2 = pts2[mask.ravel() == 1]
        print(f"Dopasowanych punktów: {len(pts1)}")
        
        # Macierz istotna
        E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC)
        if E is None:
            print(f"Nie można obliczyć macierzy istotnej. Pomijam.")
            continue
        
        print(f"Fundamental Matrix:\n{F}")
        print(f"Essential Matrix:\n{E}")
        
        # Odzyskaj pozę
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
        print(f"Recovered Rotation:\n{R}")
        print(f"Recovered Translation:\n{t}")
        
        # Macierze projekcji
        P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = K @ np.hstack([R, t])
        
        # Konwertuj na float32
        pts1_inliers = pts1[mask.ravel() > 0].astype(np.float32)
        pts2_inliers = pts2[mask.ravel() > 0].astype(np.float32)
        
        if len(pts1_inliers) < 4:
            print(f"Za mało inlierów. Pomijam.")
            continue
        
        # Triangulacja
        pts4D_hom = cv2.triangulatePoints(P1, P2, pts1_inliers.T, pts2_inliers.T)
        pts3D = (pts4D_hom[:3, :] / pts4D_hom[3, :]).T
        
        print(f"3D Points: {pts3D.shape}")
        
        # Zapisz wyniki
        all_rotations.append(R)
        all_translations.append(t.flatten())
        all_points_3d.append(pts3D)
    
    print(f"\n\nZakończyłem. Liczba przetworzonych par: {len(all_rotations)}")
    
    return all_rotations, all_translations, all_points_3d

if __name__ == "__main__":
    video_file = "WIN_20260427_15_35_38_Pro.mp4"
    
    # Wczytaj wyselekcjonowane klatki
    print("Wczytywanie klatek z pliku...")
    frames = select_frames(video_file, frame_gap=20, max_frames=100)  # Co 5-ta klatka, max 100 klatek (99 par)
    print(f"Wczytano {len(frames)} wyselekcjonowanych klatek\n")
    
    # Oblicz macierz kamery z pierwszej klatki
    K, dist = get_camera_matrix_simple(video_file)
    
    # Uruchom pipeline
    rotations, translations, points_3d = pipeline(frames, K, dist)

    # Opcjonalnie: wizualizuj wyniki
    if len(points_3d) > 0:
        combined_points = np.vstack(points_3d)
        print(f"Łącznie punktów 3D: {combined_points.shape}")
        o3d_visualization(combined_points, rotations, translations)
    
