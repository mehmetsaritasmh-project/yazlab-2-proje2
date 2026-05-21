# src/preprocessor.py
import numpy as np

class StandardScaler:
    def __init__(self):
        """
        Scikit-learn bağımlılığını azaltmak ve boru hattında 
        veri sızıntısını (Data Leakage) sıfıra indirmek için 
        tamamen izole ve sızıntısız ölçeklendirme sınıfı.
        """
        self.mean_ = None
        self.scale_ = None

    def fit(self, X: np.ndarray):
        """
        Sadece eğitim (Train) verisinin istatistiksel özetini çıkartır.
        """
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        # Sıfıra bölme hatasını engellemek için koruma bariyeri (Epsilon)
        self.scale_ = np.where(self.scale_ == 0, 1e-8, self.scale_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Eğitim verisinden öğrenilen parametrelerle gelen veriyi ölçeklendirir.
        """
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Ön işlemci önce fit() fonksiyonu ile eğitilmelidir!")
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Eğitim verisinde hem öğrenir hem de dönüştürür.
        """
        return self.fit(X).transform(X)


def add_gaussian_noise(data: np.ndarray, std_dev: float, seed: int = 2026) -> np.ndarray:
    """
    Hocanın istediği gürültü senaryosu için zaman serisi sinyaline 
    orijinal sinyal karakteristiğini bozmadan kontrollü Gaussian gürültüsü ekler.
    """
    np.random.seed(seed)
    noise = np.random.normal(loc=0.0, scale=std_dev, size=data.shape)
    return data + noise