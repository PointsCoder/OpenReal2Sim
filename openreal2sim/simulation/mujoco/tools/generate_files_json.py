#!/usr/bin/env python3
"""
Generate files.json for each demo by scanning asset directories.
Run this before dev/build to ensure files.json is up-to-date.
"""

import json
import os
from pathlib import Path


def generate_files_json(demo_path: Path) -> dict:
    """Scan demo directory and generate files.json manifest."""
    assets = []

    # Scan all subdirectories for asset files
    for root, dirs, files in os.walk(demo_path):
        # Skip if this is the demo root (we want subdirectories only)
        if Path(root) == demo_path:
            continue

        for file in sorted(files):
            # Skip metadata and config files
            if file.endswith(('.json', '.xml')):
                continue

            # Include all other files as binary assets
            rel_path = Path(root).relative_to(demo_path) / file
            assets.append({
                "path": str(rel_path).replace('\\', '/'),
                "type": "binary"
            })

    return {"assets": assets}


def main():
    demos_dir = Path(__file__).parent.parent / "public" / "demos"

    if not demos_dir.exists():
        print(f"Error: demos directory not found at {demos_dir}")
        return

    # Process each demo folder
    for demo_path in sorted(demos_dir.iterdir()):
        if not demo_path.is_dir():
            continue

        demo_name = demo_path.name
        print(f"Processing demo: {demo_name}")

        files_json = generate_files_json(demo_path)
        output_path = demo_path / "files.json"

        with open(output_path, 'w') as f:
            json.dump(files_json, f, indent=2)

        print(f"  ✓ Generated {output_path} ({len(files_json['assets'])} assets)")

    print("\n✓ All files.json generated successfully")


if __name__ == "__main__":
    main()
