"""Quick script to analyze which frames are actually matching."""
import sqlite3
from pathlib import Path

def analyze_matches(db_path: Path, top_n: int = 20):
    """Show top matching image pairs."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Get image names
    cursor.execute("SELECT image_id, name FROM images ORDER BY image_id")
    image_map = {row[0]: row[1] for row in cursor.fetchall()}

    # Get top matches
    cursor.execute("""
        SELECT pair_id, rows
        FROM two_view_geometries
        WHERE rows > 0
        ORDER BY rows DESC
        LIMIT ?
    """, (top_n,))

    print(f"\nTop {top_n} Image Pairs with Most Matches:")
    print("="*80)
    print(f"{'Image 1':<25} {'Image 2':<25} {'Matches':>10}")
    print("="*80)

    for pair_id, match_count in cursor.fetchall():
        # Decode pair_id
        image_id2 = pair_id % 2147483647
        image_id1 = (pair_id - image_id2) // 2147483647

        img1_name = image_map.get(image_id1, f"Unknown({image_id1})")
        img2_name = image_map.get(image_id2, f"Unknown({image_id2})")

        # Extract frame numbers
        try:
            frame1 = int(img1_name.split('_')[1].split('.')[0])
            frame2 = int(img2_name.split('_')[1].split('.')[0])
            gap = abs(frame2 - frame1)
            print(f"{img1_name:<25} {img2_name:<25} {match_count:>10}  (gap: {gap})")
        except:
            print(f"{img1_name:<25} {img2_name:<25} {match_count:>10}")

    conn.close()

if __name__ == "__main__":
    print("\n1-Visit Reconstruction:")
    analyze_matches(Path("results/reconstruction_1_visits/database.db"))

    print("\n\n2-Visit Reconstruction:")
    analyze_matches(Path("results/reconstruction_2_visits/database.db"))

    print("\n\n3-Visit Reconstruction:")
    analyze_matches(Path("results/reconstruction_3_visits/database.db"))
