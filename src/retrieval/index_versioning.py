import hashlib
import json
import os
from datetime import datetime, timezone

def sha256_file(path: str) -> str:
    """
    Dosyanın SHA-256 özetini 1MB'lık parçalar halinde hesaplar.
    """
    sha256_hash = hashlib.sha256()
    with open(path, "rb") as f:
        for byte_block in iter(lambda: f.read(1048576), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

class IndexVersion:
    """
    Index dosyalarının versiyon kontrolü ve bütünlük doğrulamasını yönetir.
    """

    @staticmethod
    def write(index_path: str, model_name: str, n_items: int, dimension: int, index_type: str = "IVFFlat") -> str:
        """
        Index dosyasının yanına bir JSON manifest dosyası yazar.
        Manifest dosyasının yolunu döndürür.
        """
        manifest_path = os.path.splitext(index_path)[0] + ".json"
        file_hash = sha256_file(index_path)
        
        manifest_data = {
            "model_name": model_name,
            "n_items": n_items,
            "dimension": dimension,
            "index_type": index_type,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "sha256": file_hash,
            "schema_version": 1
        }
        
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=4, ensure_ascii=False)
            
        return manifest_path

    @staticmethod
    def verify(index_path: str) -> bool:
        """
        Index dosyasının yanındaki manifest dosyasını okuyarak SHA-256 doğrulamasını yapar.
        Doğrulanırsa True döndürür, aksi takdirde ValueError fırlatır.
        """
        manifest_path = os.path.splitext(index_path)[0] + ".json"
        
        if not os.path.exists(manifest_path):
            raise ValueError(f"Manifest dosyası bulunamadı: {manifest_path}")
            
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
            
        expected_hash = manifest_data.get("sha256")
        if not expected_hash:
            raise ValueError(f"Manifest dosyasında 'sha256' alanı bulunamadı: {manifest_path}")
            
        actual_hash = sha256_file(index_path)
        
        if expected_hash != actual_hash:
            raise ValueError(f"SHA-256 doğrulaması başarısız! Beklenen: {expected_hash}, Bulunan: {actual_hash}")
            
        return True
