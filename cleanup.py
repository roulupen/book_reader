#!/usr/bin/env python3
"""
Cleanup script for the Book Reader RAG application
Removes any temporary files and orphaned data
"""
import os
import shutil
from pathlib import Path
from app.config import settings

def cleanup_temp_files():
    """Remove all temporary files from upload directories"""
    print("🧹 Cleaning up temporary files...")
    
    cleaned_count = 0
    
    # Clean uploads folder
    upload_folder = Path(settings.UPLOAD_FOLDER)
    if upload_folder.exists():
        for file_path in upload_folder.iterdir():
            if file_path.is_file():
                file_path.unlink()
                print(f"   Deleted: {file_path}")
                cleaned_count += 1
    
    # Clean data folder (since files should be deleted after indexing)
    data_folder = Path(settings.DATA_FOLDER)
    if data_folder.exists():
        for file_path in data_folder.iterdir():
            if file_path.is_file():
                file_path.unlink()
                print(f"   Deleted: {file_path}")
                cleaned_count += 1
    
    print(f"✅ Cleaned up {cleaned_count} files")
    return cleaned_count

def cleanup_empty_directories():
    """Remove empty directories"""
    print("📁 Checking for empty directories...")
    
    folders_to_check = [
        Path(settings.UPLOAD_FOLDER),
        Path(settings.DATA_FOLDER)
    ]
    
    for folder in folders_to_check:
        if folder.exists() and not any(folder.iterdir()):
            print(f"   Empty directory: {folder}")
    
    print("✅ Directory check complete")

def show_disk_usage():
    """Show disk usage of application directories"""
    print("💾 Disk usage:")
    
    folders = [
        ("Uploads", settings.UPLOAD_FOLDER),
        ("Data", settings.DATA_FOLDER),
        ("ChromaDB", settings.CHROMA_PERSIST_DIRECTORY)
    ]
    
    total_size = 0
    
    for name, folder_path in folders:
        folder = Path(folder_path)
        if folder.exists():
            size = sum(f.stat().st_size for f in folder.rglob('*') if f.is_file())
            size_mb = size / (1024 * 1024)
            print(f"   {name}: {size_mb:.2f} MB")
            total_size += size
        else:
            print(f"   {name}: Directory doesn't exist")
    
    total_mb = total_size / (1024 * 1024)
    print(f"   Total: {total_mb:.2f} MB")

def main():
    """Main cleanup function"""
    print("🚀 Book Reader RAG Cleanup Tool")
    print("=" * 40)
    
    # Show current disk usage
    show_disk_usage()
    print()
    
    # Clean temporary files
    cleaned = cleanup_temp_files()
    print()
    
    # Check directories
    cleanup_empty_directories()
    print()
    
    # Show final disk usage
    if cleaned > 0:
        print("📊 Disk usage after cleanup:")
        show_disk_usage()
        print()
    
    print("✨ Cleanup complete!")
    print("\nNote: Files are automatically deleted after indexing.")
    print("This script only cleans up any orphaned files.")

if __name__ == "__main__":
    main()
