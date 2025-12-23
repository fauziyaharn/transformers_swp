"""
Recommendation Engine Module
Modul untuk rekomendasi menggunakan K-Means Clustering
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from typing import List, Dict, Any
import json


class RecommendationEngine:
    """Class untuk sistem rekomendasi dengan K-Means Clustering"""
    
    def __init__(self, n_clusters=3):
        """
        Inisialisasi recommendation engine
        
        Args:
            n_clusters: Jumlah cluster untuk K-Means (default: 3)
        """
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.kmeans = None
    
    def prepare_features(self, items: List[Dict], slots: Dict) -> np.ndarray:
        """
        Mempersiapkan fitur untuk clustering
        
        Args:
            items: List of dict berisi data items dari database
            slots: Dict berisi slot yang diekstrak dari AI
            
        Returns:
            Numpy array berisi fitur yang sudah dinormalisasi
        """
        if not items:
            return np.array([])
        
        features = []
        
        for item in items:
            feature_vector = []

            # Feature 1: Harga (raw)
            price = item.get('price', 0) or 0
            feature_vector.append(price)

            # Feature 2: Relevance tema (0/1)
            tema = slots.get('tema', '') or ''
            relevance_tema = 0
            if tema:
                tema_lower = tema.lower() if isinstance(tema, str) else ''
                item_text = f"{item.get('name', '') or ''} {item.get('description', '') or ''} {item.get('vendor', '') or ''} {item.get('category_name', '') or ''}".lower()
                if tema_lower in item_text:
                    relevance_tema = 1
            feature_vector.append(relevance_tema)

            # Feature 3: Relevance lokasi (improve: check explicit city fields and text)
            lokasi = slots.get('lokasi', '') or ''
            relevance_lokasi = 0
            if lokasi:
                lokasi_lower = lokasi.lower() if isinstance(lokasi, str) else ''
                # Prefer explicit fields
                item_locs = []
                for k in ['city', 'lokasi', 'location', 'kota']:
                    v = item.get(k)
                    if v:
                        item_locs.append(str(v).lower())
                # fallback to vendor/description/category
                item_text = f"{item.get('vendor', '') or ''} {item.get('description', '') or ''} {item.get('category_name', '') or ''}".lower()
                if any(lokasi_lower in l for l in item_locs) or lokasi_lower in item_text:
                    relevance_lokasi = 1
            feature_vector.append(relevance_lokasi)

            # Feature 4: Budget fit score (0..1) — closer to user budget gives higher score
            budget_min = slots.get('budget_min') if slots.get('budget_min') is not None else 0
            budget_max = slots.get('budget_max') if slots.get('budget_max') is not None else float('inf')

            budget_fit = 0.0
            if budget_min and budget_max and budget_min <= budget_max and budget_min <= price <= budget_max:
                budget_fit = 1.0
            else:
                # compute distance to nearest bound (if bounds exist)
                nearest = None
                if budget_min and price < budget_min:
                    nearest = budget_min
                elif budget_max and price > budget_max and budget_max != float('inf'):
                    nearest = budget_max
                if nearest:
                    # normalized closeness (higher = better)
                    diff = abs(price - nearest)
                    denom = max(nearest, price, 1)
                    budget_fit = max(0.0, 1.0 - (diff / denom))
                else:
                    # if no bounds, give neutral score based on price magnitude
                    budget_fit = 0.5

            feature_vector.append(float(budget_fit))

            features.append(feature_vector)
        
        # Konversi ke numpy array
        features_array = np.array(features)
        
        # Normalisasi fitur
        if len(features_array) > 0:
            features_normalized = self.scaler.fit_transform(features_array)
            return features_normalized
        
        return features_array
    
    def calculate_relevance_score(self, item: Dict, slots: Dict) -> float:
        """
        Menghitung skor relevansi item terhadap slot yang diminta
        
        Args:
            item: Dict berisi data item
            slots: Dict berisi slot dari AI
            
        Returns:
            Skor relevansi (0-1)
        """
        score = 0.0
        total_checks = 0

        # Check tema
        tema = slots.get('tema', '') or ''
        if tema:
            total_checks += 1
            tema_lower = tema.lower() if isinstance(tema, str) else ''
            item_text = f"{item.get('name', '') or ''} {item.get('description', '') or ''} {item.get('vendor', '') or ''} {item.get('category_name', '') or ''}".lower()
            if tema_lower in item_text:
                score += 1.0

        # Check lokasi (strong match if explicit city field)
        lokasi = slots.get('lokasi', '') or ''
        if lokasi:
            total_checks += 1
            lokasi_lower = lokasi.lower() if isinstance(lokasi, str) else ''
            # explicit keys
            item_locs = []
            for k in ['city', 'lokasi', 'location', 'kota']:
                v = item.get(k)
                if v:
                    item_locs.append(str(v).lower())
            item_text = f"{item.get('vendor', '') or ''} {item.get('description', '') or ''} {item.get('category_name', '') or ''}".lower()
            if any(lokasi_lower == l or lokasi_lower in l for l in item_locs):
                score += 1.0
            elif lokasi_lower in item_text:
                score += 0.7

        # Check budget (with tolerance and closeness)
        budget_min = slots.get('budget_min') if slots.get('budget_min') is not None else 0
        budget_max = slots.get('budget_max') if slots.get('budget_max') is not None else float('inf')
        if (budget_min and budget_min > 0) or (budget_max and budget_max != float('inf')):
            total_checks += 1
            price = item.get('price', 0) or 0

            # exact within range
            if budget_min and budget_max and budget_min <= price <= budget_max:
                score += 1.0
            else:
                # tolerance window 20%
                tol = 0.2
                min_tol = budget_min * (1 - tol) if budget_min else 0
                max_tol = budget_max * (1 + tol) if budget_max and budget_max != float('inf') else float('inf')
                if min_tol <= price <= max_tol:
                    # closer to bounds -> higher score
                    # normalized closeness to nearest bound
                    nearest = budget_min if price < budget_min else budget_max if price > budget_max and budget_max != float('inf') else None
                    if nearest:
                        diff = abs(price - nearest)
                        denom = max(nearest, price, 1)
                        closeness = max(0.0, 1.0 - (diff / denom))
                        score += 0.6 + 0.4 * closeness
                    else:
                        score += 0.6
                elif price < budget_min:
                    score += 0.3
                else:
                    score += 0.2

        return score / total_checks if total_checks > 0 else 0
    
    def cluster_items(self, items: List[Dict], slots: Dict) -> Dict[str, Any]:
        """
        Melakukan clustering pada items dan mengembalikan rekomendasi terbaik
        
        Args:
            items: List of dict berisi data items dari database
            slots: Dict berisi slot dari AI
            
        Returns:
            Dict berisi hasil clustering dan rekomendasi
        """
        if not items or len(items) == 0:
            return {
                "total_items": 0,
                "clusters": [],
                "recommendations": [],
                "message": "Tidak ada item yang ditemukan"
            }
        
        # Jika item terlalu sedikit, tidak perlu clustering
        if len(items) < self.n_clusters:
            # Urutkan berdasarkan relevance score
            for item in items:
                item['relevance_score'] = self.calculate_relevance_score(item, slots)
            
            items_sorted = sorted(items, key=lambda x: x['relevance_score'], reverse=True)
            
            return {
                "total_items": len(items),
                "clusters": [{"cluster_id": 0, "size": len(items)}],
                "recommendations": items_sorted[:10],  # Top 10
                "message": f"Item terlalu sedikit untuk clustering, menampilkan {len(items_sorted[:10])} rekomendasi terbaik"
            }
        
        # Prepare features untuk clustering
        features = self.prepare_features(items, slots)
        
        if len(features) == 0:
            return {
                "total_items": 0,
                "clusters": [],
                "recommendations": [],
                "message": "Gagal mempersiapkan fitur untuk clustering"
            }
        
        # Lakukan K-Means clustering
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        cluster_labels = self.kmeans.fit_predict(features)
        
        # Tambahkan cluster label ke setiap item
        for i, item in enumerate(items):
            item['cluster'] = int(cluster_labels[i])
            item['relevance_score'] = self.calculate_relevance_score(item, slots)
        
        # Analisis cluster
        cluster_info = []
        for cluster_id in range(self.n_clusters):
            cluster_items = [item for item in items if item['cluster'] == cluster_id]
            if cluster_items:
                avg_price = np.mean([item.get('price', 0) or 0 for item in cluster_items])
                avg_relevance = np.mean([item['relevance_score'] for item in cluster_items])
                
                cluster_info.append({
                    "cluster_id": cluster_id,
                    "size": len(cluster_items),
                    "avg_price": int(avg_price),
                    "avg_relevance": round(avg_relevance, 2),
                    "items": cluster_items
                })
        
        # Pilih cluster dengan relevance tertinggi
        best_cluster = max(cluster_info, key=lambda x: x['avg_relevance'])
        
        # Urutkan item dalam cluster terbaik berdasarkan relevance score
        recommendations = sorted(best_cluster['items'], 
                               key=lambda x: (x['relevance_score'], -x.get('price', 0)), 
                               reverse=True)[:10]  # Top 10 recommendations
        
        return {
            "total_items": len(items),
            "n_clusters": self.n_clusters,
            "clusters": [
                {
                    "cluster_id": c['cluster_id'],
                    "size": c['size'],
                    "avg_price": c['avg_price'],
                    "avg_relevance": c['avg_relevance']
                } for c in cluster_info
            ],
            "best_cluster_id": best_cluster['cluster_id'],
            "recommendations": recommendations,
            "message": f"Ditemukan {len(recommendations)} rekomendasi terbaik dari cluster #{best_cluster['cluster_id']}"
        }
    
    def format_recommendation_output(self, result: Dict, slots: Dict) -> str:
        """
        Format hasil rekomendasi untuk ditampilkan
        
        Args:
            result: Dict hasil dari cluster_items
            slots: Dict berisi slot dari AI
            
        Returns:
            String berisi output yang terformat
        """
        output = []
        output.append("\n" + "="*80)
        output.append("HASIL PENCARIAN DAN CLUSTERING")
        output.append("="*80)
        
        # Tampilkan filter yang digunakan
        output.append("\n📋 FILTER PENCARIAN:")
        if slots.get('tema'):
            output.append(f"   • Tema: {slots['tema']}")
        if slots.get('lokasi'):
            output.append(f"   • Lokasi: {slots['lokasi']}")
        if slots.get('budget_min') or slots.get('budget_max'):
            budget_str = ""
            if slots.get('budget_min'):
                budget_str += f"Rp {slots['budget_min']:,}"
            if slots.get('budget_max'):
                if budget_str:
                    budget_str += f" - Rp {slots['budget_max']:,}"
                else:
                    budget_str += f"max Rp {slots['budget_max']:,}"
            output.append(f"   • Budget: {budget_str}")
        if slots.get('jumlah_tamu'):
            output.append(f"   • Jumlah Tamu: {slots['jumlah_tamu']} orang")
        
        # Informasi clustering
        output.append(f"\n📊 INFORMASI CLUSTERING:")
        output.append(f"   • Total item ditemukan: {result['total_items']}")
        output.append(f"   • Jumlah cluster: {len(result['clusters'])}")
        
        if result['clusters']:
            output.append(f"\n   Distribusi Cluster:")
            for cluster in result['clusters']:
                output.append(f"      - Cluster #{cluster['cluster_id']}: "
                            f"{cluster['size']} item, "
                            f"Avg Price: Rp {cluster['avg_price']:,}, "
                            f"Avg Relevance: {cluster['avg_relevance']}")
        
        if 'best_cluster_id' in result:
            output.append(f"\n   ✓ Cluster terbaik: #{result['best_cluster_id']}")
        
        # Rekomendasi
        output.append(f"\n🎯 REKOMENDASI TERBAIK ({len(result['recommendations'])} item):")
        output.append("-"*80)
        
        for idx, item in enumerate(result['recommendations'], 1):
            output.append(f"\n{idx}. {item.get('name', 'N/A')}")
            output.append(f"   Vendor: {item.get('vendor', 'N/A')}")
            output.append(f"   Kategori: {item.get('category_name', 'N/A')}")
            output.append(f"   Harga: Rp {item.get('price', 0):,}")
            output.append(f"   Relevance Score: {item.get('relevance_score', 0):.2f}")
            if item.get('description'):
                desc = item['description'][:100] + "..." if len(item['description']) > 100 else item['description']
                output.append(f"   Deskripsi: {desc}")
            if item.get('cluster') is not None:
                output.append(f"   Cluster: #{item['cluster']}")
        
        output.append("\n" + "="*80)
        output.append(result.get('message', ''))
        output.append("="*80 + "\n")
        
        return "\n".join(output)


if __name__ == "__main__":
    # Test recommendation engine
    print("=== Testing Recommendation Engine ===")
    
    # Mock data untuk testing
    mock_items = [
        {"id": "1", "name": "Paket MUA Adat Jawa", "vendor": "Jakarta Makeup", "description": "Riasan adat jawa", "price": 4000000, "category_name": "MUA"},
        {"id": "2", "name": "Venue Outdoor Sunda", "vendor": "Bandung Hall", "description": "Venue tema sunda", "price": 35000000, "category_name": "Venue"},
        {"id": "3", "name": "Catering Premium", "vendor": "Jakarta Catering", "description": "Menu lengkap", "price": 45000000, "category_name": "Catering"},
        {"id": "4", "name": "MUA Sunda Siger", "vendor": "Jakarta Beauty", "description": "Riasan sunda siger", "price": 4200000, "category_name": "MUA"},
        {"id": "5", "name": "Dekorasi Adat Sunda", "vendor": "Dekor Jakarta", "description": "Dekorasi pernikahan sunda", "price": 15000000, "category_name": "Decoration"},
    ]
    
    mock_slots = {
        "tema": "sunda",
        "lokasi": "jakarta",
        "budget_min": 2000000,
        "budget_max": 50000000,
        "jumlah_tamu": 500
    }
    
    engine = RecommendationEngine(n_clusters=2)
    result = engine.cluster_items(mock_items, mock_slots)
    
    print(engine.format_recommendation_output(result, mock_slots))
