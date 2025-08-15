"""
Internationalization (i18n) Support for IoT Edge Anomaly Detection.

This module provides comprehensive multi-language support including:
- Dynamic language switching
- Resource bundles for different locales
- Locale-aware formatting (dates, numbers, currencies)
- Right-to-left (RTL) language support
- Cultural adaptation for different regions
"""
import json
import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import datetime
import re
from functools import lru_cache

logger = logging.getLogger(__name__)


class SupportedLanguage(Enum):
    """Supported languages with their codes."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE_SIMPLIFIED = "zh"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    ARABIC = "ar"
    HINDI = "hi"


class TextDirection(Enum):
    """Text direction for different languages."""
    LTR = "ltr"  # Left to Right
    RTL = "rtl"  # Right to Left


@dataclass
class LocaleInfo:
    """Information about a specific locale."""
    language_code: str
    language_name: str
    native_name: str
    direction: TextDirection
    decimal_separator: str = "."
    thousands_separator: str = ","
    date_format: str = "%Y-%m-%d"
    time_format: str = "%H:%M:%S"
    currency_symbol: str = "$"
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.language_code in ["ar", "he", "fa"]:  # Arabic, Hebrew, Persian
            self.direction = TextDirection.RTL


class LocalizationManager:
    """
    Comprehensive localization manager for multi-language support.
    
    Features:
    - Dynamic language switching
    - Hierarchical message resolution
    - Pluralization support
    - Parameter interpolation
    - Locale-specific formatting
    """
    
    def __init__(self, default_language: str = "en"):
        """Initialize localization manager."""
        self.default_language = default_language
        self.current_language = default_language
        
        # Language resources
        self.message_bundles: Dict[str, Dict[str, Any]] = {}
        self.locale_info: Dict[str, LocaleInfo] = {}
        
        # Initialize supported locales
        self._initialize_locales()
        
        # Load default message bundles
        self._load_default_messages()
        
        logger.info(f"Localization manager initialized with default language: {default_language}")
    
    def _initialize_locales(self) -> None:
        """Initialize supported locale information."""
        self.locale_info = {
            "en": LocaleInfo("en", "English", "English", TextDirection.LTR, ".", ",", "%m/%d/%Y", "%I:%M %p", "$"),
            "es": LocaleInfo("es", "Spanish", "Español", TextDirection.LTR, ",", ".", "%d/%m/%Y", "%H:%M", "€"),
            "fr": LocaleInfo("fr", "French", "Français", TextDirection.LTR, ",", " ", "%d/%m/%Y", "%H:%M", "€"),
            "de": LocaleInfo("de", "German", "Deutsch", TextDirection.LTR, ",", ".", "%d.%m.%Y", "%H:%M", "€"),
            "ja": LocaleInfo("ja", "Japanese", "日本語", TextDirection.LTR, ".", ",", "%Y/%m/%d", "%H:%M", "¥"),
            "zh": LocaleInfo("zh", "Chinese", "中文", TextDirection.LTR, ".", ",", "%Y-%m-%d", "%H:%M", "¥"),
            "pt": LocaleInfo("pt", "Portuguese", "Português", TextDirection.LTR, ",", ".", "%d/%m/%Y", "%H:%M", "R$"),
            "ru": LocaleInfo("ru", "Russian", "Русский", TextDirection.LTR, ",", " ", "%d.%m.%Y", "%H:%M", "₽"),
            "ar": LocaleInfo("ar", "Arabic", "العربية", TextDirection.RTL, ".", ",", "%d/%m/%Y", "%H:%M", "ريال"),
            "hi": LocaleInfo("hi", "Hindi", "हिन्दी", TextDirection.LTR, ".", ",", "%d/%m/%Y", "%H:%M", "₹")
        }
    
    def _load_default_messages(self) -> None:
        """Load default message bundles for all supported languages."""
        
        # English (base language)
        self.message_bundles["en"] = {
            # System messages
            "system.startup": "IoT Edge Anomaly Detection System Starting",
            "system.shutdown": "System Shutting Down",
            "system.ready": "System Ready",
            "system.error": "System Error",
            
            # Anomaly detection messages
            "anomaly.detected": "Anomaly Detected",
            "anomaly.severity.low": "Low Severity",
            "anomaly.severity.medium": "Medium Severity",
            "anomaly.severity.high": "High Severity",
            "anomaly.severity.critical": "Critical Severity",
            
            # Model messages
            "model.loading": "Loading Model",
            "model.loaded": "Model Loaded Successfully",
            "model.training": "Training Model",
            "model.inference": "Running Inference",
            "model.error": "Model Error",
            
            # Security messages
            "security.threat.detected": "Security Threat Detected",
            "security.access.denied": "Access Denied",
            "security.authentication.failed": "Authentication Failed",
            "security.validation.error": "Input Validation Error",
            
            # Performance messages
            "performance.optimization": "Optimizing Performance",
            "performance.cache.hit": "Cache Hit",
            "performance.cache.miss": "Cache Miss",
            "performance.scaling.up": "Scaling Up",
            "performance.scaling.down": "Scaling Down",
            
            # Error messages
            "error.connection": "Connection Error",
            "error.timeout": "Operation Timeout",
            "error.insufficient.memory": "Insufficient Memory",
            "error.invalid.input": "Invalid Input",
            "error.configuration": "Configuration Error",
            
            # Units and measurements
            "unit.milliseconds": "ms",
            "unit.seconds": "s",
            "unit.minutes": "min",
            "unit.hours": "h",
            "unit.bytes": "B",
            "unit.kilobytes": "KB",
            "unit.megabytes": "MB",
            "unit.gigabytes": "GB",
            "unit.percent": "%",
            
            # UI elements
            "ui.start": "Start",
            "ui.stop": "Stop",
            "ui.restart": "Restart",
            "ui.configure": "Configure",
            "ui.status": "Status",
            "ui.logs": "Logs",
            "ui.metrics": "Metrics",
            "ui.help": "Help"
        }
        
        # Spanish translations
        self.message_bundles["es"] = {
            "system.startup": "Sistema de Detección de Anomalías IoT Edge Iniciando",
            "system.shutdown": "Sistema Cerrando",
            "system.ready": "Sistema Listo",
            "system.error": "Error del Sistema",
            
            "anomaly.detected": "Anomalía Detectada",
            "anomaly.severity.low": "Severidad Baja",
            "anomaly.severity.medium": "Severidad Media",
            "anomaly.severity.high": "Severidad Alta",
            "anomaly.severity.critical": "Severidad Crítica",
            
            "model.loading": "Cargando Modelo",
            "model.loaded": "Modelo Cargado Exitosamente",
            "model.training": "Entrenando Modelo",
            "model.inference": "Ejecutando Inferencia",
            "model.error": "Error del Modelo",
            
            "security.threat.detected": "Amenaza de Seguridad Detectada",
            "security.access.denied": "Acceso Denegado",
            "security.authentication.failed": "Autenticación Fallida",
            "security.validation.error": "Error de Validación de Entrada",
            
            "performance.optimization": "Optimizando Rendimiento",
            "performance.cache.hit": "Acierto de Caché",
            "performance.cache.miss": "Fallo de Caché",
            "performance.scaling.up": "Escalando Hacia Arriba",
            "performance.scaling.down": "Escalando Hacia Abajo",
            
            "error.connection": "Error de Conexión",
            "error.timeout": "Tiempo de Espera Agotado",
            "error.insufficient.memory": "Memoria Insuficiente",
            "error.invalid.input": "Entrada Inválida",
            "error.configuration": "Error de Configuración",
            
            "ui.start": "Iniciar",
            "ui.stop": "Detener",
            "ui.restart": "Reiniciar",
            "ui.configure": "Configurar",
            "ui.status": "Estado",
            "ui.logs": "Registros",
            "ui.metrics": "Métricas",
            "ui.help": "Ayuda"
        }
        
        # French translations
        self.message_bundles["fr"] = {
            "system.startup": "Système de Détection d'Anomalies IoT Edge Démarrage",
            "system.shutdown": "Arrêt du Système",
            "system.ready": "Système Prêt",
            "system.error": "Erreur Système",
            
            "anomaly.detected": "Anomalie Détectée",
            "anomaly.severity.low": "Gravité Faible",
            "anomaly.severity.medium": "Gravité Moyenne",
            "anomaly.severity.high": "Gravité Élevée",
            "anomaly.severity.critical": "Gravité Critique",
            
            "model.loading": "Chargement du Modèle",
            "model.loaded": "Modèle Chargé avec Succès",
            "model.training": "Entraînement du Modèle",
            "model.inference": "Exécution de l'Inférence",
            "model.error": "Erreur de Modèle",
            
            "security.threat.detected": "Menace de Sécurité Détectée",
            "security.access.denied": "Accès Refusé",
            "security.authentication.failed": "Échec de l'Authentification",
            "security.validation.error": "Erreur de Validation d'Entrée",
            
            "performance.optimization": "Optimisation des Performances",
            "performance.cache.hit": "Succès du Cache",
            "performance.cache.miss": "Échec du Cache",
            "performance.scaling.up": "Montée en Charge",
            "performance.scaling.down": "Réduction de Charge",
            
            "ui.start": "Démarrer",
            "ui.stop": "Arrêter",
            "ui.restart": "Redémarrer",
            "ui.configure": "Configurer",
            "ui.status": "État",
            "ui.logs": "Journaux",
            "ui.metrics": "Métriques",
            "ui.help": "Aide"
        }
        
        # German translations
        self.message_bundles["de"] = {
            "system.startup": "IoT Edge Anomalieerkennung System Startet",
            "system.shutdown": "System Herunterfahren",
            "system.ready": "System Bereit",
            "system.error": "Systemfehler",
            
            "anomaly.detected": "Anomalie Erkannt",
            "anomaly.severity.low": "Niedrige Schwere",
            "anomaly.severity.medium": "Mittlere Schwere",
            "anomaly.severity.high": "Hohe Schwere",
            "anomaly.severity.critical": "Kritische Schwere",
            
            "model.loading": "Modell Laden",
            "model.loaded": "Modell Erfolgreich Geladen",
            "model.training": "Modell Trainieren",
            "model.inference": "Inferenz Ausführen",
            "model.error": "Modellfehler",
            
            "ui.start": "Starten",
            "ui.stop": "Stoppen",
            "ui.restart": "Neustarten",
            "ui.configure": "Konfigurieren",
            "ui.status": "Status",
            "ui.logs": "Protokolle",
            "ui.metrics": "Metriken",
            "ui.help": "Hilfe"
        }
        
        # Japanese translations
        self.message_bundles["ja"] = {
            "system.startup": "IoTエッジ異常検知システム起動中",
            "system.shutdown": "システム終了",
            "system.ready": "システム準備完了",
            "system.error": "システムエラー",
            
            "anomaly.detected": "異常検知",
            "anomaly.severity.low": "軽度",
            "anomaly.severity.medium": "中度",
            "anomaly.severity.high": "重度",
            "anomaly.severity.critical": "重大",
            
            "model.loading": "モデル読み込み中",
            "model.loaded": "モデル読み込み完了",
            "model.training": "モデル訓練中",
            "model.inference": "推論実行中",
            "model.error": "モデルエラー",
            
            "ui.start": "開始",
            "ui.stop": "停止",
            "ui.restart": "再起動",
            "ui.configure": "設定",
            "ui.status": "状態",
            "ui.logs": "ログ",
            "ui.metrics": "メトリクス",
            "ui.help": "ヘルプ"
        }
        
        # Chinese translations
        self.message_bundles["zh"] = {
            "system.startup": "IoT边缘异常检测系统启动中",
            "system.shutdown": "系统关闭",
            "system.ready": "系统就绪",
            "system.error": "系统错误",
            
            "anomaly.detected": "检测到异常",
            "anomaly.severity.low": "低严重性",
            "anomaly.severity.medium": "中等严重性",
            "anomaly.severity.high": "高严重性",
            "anomaly.severity.critical": "严重",
            
            "model.loading": "模型加载中",
            "model.loaded": "模型加载成功",
            "model.training": "模型训练中",
            "model.inference": "运行推理",
            "model.error": "模型错误",
            
            "ui.start": "开始",
            "ui.stop": "停止",
            "ui.restart": "重启",
            "ui.configure": "配置",
            "ui.status": "状态",
            "ui.logs": "日志",
            "ui.metrics": "指标",
            "ui.help": "帮助"
        }
        
        # Arabic translations
        self.message_bundles["ar"] = {
            "system.startup": "نظام كشف الشذوذ IoT Edge يبدأ التشغيل",
            "system.shutdown": "إيقاف تشغيل النظام",
            "system.ready": "النظام جاهز",
            "system.error": "خطأ في النظام",
            
            "anomaly.detected": "تم اكتشاف شذوذ",
            "anomaly.severity.low": "خطورة منخفضة",
            "anomaly.severity.medium": "خطورة متوسطة",
            "anomaly.severity.high": "خطورة عالية",
            "anomaly.severity.critical": "خطورة حرجة",
            
            "ui.start": "بدء",
            "ui.stop": "توقف",
            "ui.restart": "إعادة تشغيل",
            "ui.configure": "تكوين",
            "ui.status": "الحالة",
            "ui.logs": "السجلات",
            "ui.metrics": "المقاييس",
            "ui.help": "مساعدة"
        }
    
    def set_language(self, language_code: str) -> bool:
        """
        Set the current language.
        
        Args:
            language_code: Language code (e.g., 'en', 'es', 'fr')
            
        Returns:
            True if language was set successfully
        """
        if language_code in self.locale_info:
            self.current_language = language_code
            logger.info(f"Language changed to: {language_code}")
            return True
        else:
            logger.warning(f"Unsupported language: {language_code}")
            return False
    
    def get_message(self, key: str, language: Optional[str] = None, **kwargs) -> str:
        """
        Get localized message for the given key.
        
        Args:
            key: Message key (e.g., 'system.startup')
            language: Optional language override
            **kwargs: Parameters for message interpolation
            
        Returns:
            Localized message string
        """
        lang = language or self.current_language
        
        # Try current language first
        if lang in self.message_bundles and key in self.message_bundles[lang]:
            message = self.message_bundles[lang][key]
        # Fall back to default language
        elif self.default_language in self.message_bundles and key in self.message_bundles[self.default_language]:
            message = self.message_bundles[self.default_language][key]
            logger.debug(f"Falling back to default language for key: {key}")
        # Fall back to key itself
        else:
            message = key
            logger.warning(f"No translation found for key: {key}")
        
        # Perform parameter interpolation
        if kwargs:
            try:
                message = message.format(**kwargs)
            except (KeyError, ValueError) as e:
                logger.warning(f"Parameter interpolation failed for key {key}: {e}")
        
        return message
    
    @lru_cache(maxsize=100)
    def get_locale_info(self, language_code: Optional[str] = None) -> LocaleInfo:
        """Get locale information for the specified language."""
        lang = language_code or self.current_language
        return self.locale_info.get(lang, self.locale_info[self.default_language])
    
    def format_number(self, number: Union[int, float], language: Optional[str] = None) -> str:
        """Format number according to locale conventions."""
        locale = self.get_locale_info(language)
        
        # Convert to string and split on decimal point
        num_str = f"{number:.2f}" if isinstance(number, float) else str(number)
        
        if '.' in num_str:
            integer_part, decimal_part = num_str.split('.')
        else:
            integer_part, decimal_part = num_str, ""
        
        # Add thousands separators
        if len(integer_part) > 3:
            # Add thousands separator every 3 digits from right
            formatted_integer = ""
            for i, digit in enumerate(reversed(integer_part)):
                if i > 0 and i % 3 == 0:
                    formatted_integer = locale.thousands_separator + formatted_integer
                formatted_integer = digit + formatted_integer
            integer_part = formatted_integer
        
        # Combine with decimal part
        if decimal_part and decimal_part != "00":
            return f"{integer_part}{locale.decimal_separator}{decimal_part}"
        else:
            return integer_part
    
    def format_currency(self, amount: Union[int, float], language: Optional[str] = None) -> str:
        """Format currency according to locale conventions."""
        locale = self.get_locale_info(language)
        formatted_number = self.format_number(amount, language)
        
        # Currency placement varies by locale
        if locale.language_code in ["en"]:
            return f"{locale.currency_symbol}{formatted_number}"
        else:
            return f"{formatted_number} {locale.currency_symbol}"
    
    def format_datetime(self, dt: datetime.datetime, language: Optional[str] = None, 
                       include_time: bool = True) -> str:
        """Format datetime according to locale conventions."""
        locale = self.get_locale_info(language)
        
        if include_time:
            format_str = f"{locale.date_format} {locale.time_format}"
        else:
            format_str = locale.date_format
        
        return dt.strftime(format_str)
    
    def format_percentage(self, value: float, language: Optional[str] = None, 
                         decimal_places: int = 1) -> str:
        """Format percentage according to locale conventions."""
        locale = self.get_locale_info(language)
        
        # Format the number
        formatted_value = f"{value:.{decimal_places}f}"
        
        # Replace decimal separator
        if locale.decimal_separator != ".":
            formatted_value = formatted_value.replace(".", locale.decimal_separator)
        
        return f"{formatted_value}%"
    
    def pluralize(self, count: int, singular_key: str, plural_key: Optional[str] = None, 
                 language: Optional[str] = None) -> str:
        """
        Get pluralized message based on count.
        
        Args:
            count: Number to determine pluralization
            singular_key: Message key for singular form
            plural_key: Message key for plural form (defaults to singular_key + ".plural")
            language: Optional language override
            
        Returns:
            Appropriately pluralized message
        """
        if plural_key is None:
            plural_key = f"{singular_key}.plural"
        
        # Simple pluralization rules (can be extended for complex languages)
        lang = language or self.current_language
        
        if lang in ["ja", "zh", "ko"]:  # Languages without plural forms
            key = singular_key
        elif count == 1:
            key = singular_key
        else:
            key = plural_key
        
        return self.get_message(key, language, count=count)
    
    def get_supported_languages(self) -> List[Dict[str, str]]:
        """Get list of supported languages with their information."""
        return [
            {
                "code": info.language_code,
                "name": info.language_name,
                "native_name": info.native_name,
                "direction": info.direction.value
            }
            for info in self.locale_info.values()
        ]
    
    def load_custom_messages(self, language_code: str, messages: Dict[str, str]) -> None:
        """Load custom message bundle for a language."""
        if language_code not in self.message_bundles:
            self.message_bundles[language_code] = {}
        
        self.message_bundles[language_code].update(messages)
        logger.info(f"Loaded {len(messages)} custom messages for language: {language_code}")
    
    def load_messages_from_file(self, file_path: Union[str, Path]) -> bool:
        """Load message bundles from JSON file."""
        try:
            file_path = Path(file_path)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                messages_data = json.load(f)
            
            for language_code, messages in messages_data.items():
                self.load_custom_messages(language_code, messages)
            
            logger.info(f"Loaded messages from file: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load messages from file {file_path}: {e}")
            return False
    
    def export_messages(self, file_path: Union[str, Path], languages: Optional[List[str]] = None) -> bool:
        """Export message bundles to JSON file."""
        try:
            file_path = Path(file_path)
            
            # Determine which languages to export
            if languages is None:
                export_data = self.message_bundles
            else:
                export_data = {
                    lang: self.message_bundles[lang] 
                    for lang in languages 
                    if lang in self.message_bundles
                }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported messages to file: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export messages to file {file_path}: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get localization manager status."""
        return {
            "current_language": self.current_language,
            "default_language": self.default_language,
            "supported_languages": len(self.locale_info),
            "loaded_message_bundles": list(self.message_bundles.keys()),
            "total_messages": {
                lang: len(messages) 
                for lang, messages in self.message_bundles.items()
            }
        }


# Global localization manager instance
_localization_manager: Optional[LocalizationManager] = None


def get_localization_manager(default_language: str = "en") -> LocalizationManager:
    """Get or create global localization manager."""
    global _localization_manager
    
    if _localization_manager is None:
        _localization_manager = LocalizationManager(default_language)
    
    return _localization_manager


def set_language(language_code: str) -> bool:
    """Set the current language globally."""
    manager = get_localization_manager()
    return manager.set_language(language_code)


def translate(key: str, language: Optional[str] = None, **kwargs) -> str:
    """
    Translate a message key to the current or specified language.
    
    Args:
        key: Message key
        language: Optional language override
        **kwargs: Parameters for message interpolation
        
    Returns:
        Translated message
    """
    manager = get_localization_manager()
    return manager.get_message(key, language, **kwargs)


def format_number(number: Union[int, float], language: Optional[str] = None) -> str:
    """Format number according to current locale."""
    manager = get_localization_manager()
    return manager.format_number(number, language)


def format_currency(amount: Union[int, float], language: Optional[str] = None) -> str:
    """Format currency according to current locale."""
    manager = get_localization_manager()
    return manager.format_currency(amount, language)


def format_datetime(dt: datetime.datetime, language: Optional[str] = None, 
                   include_time: bool = True) -> str:
    """Format datetime according to current locale."""
    manager = get_localization_manager()
    return manager.format_datetime(dt, language, include_time)


# Convenience aliases for common translations
_ = translate  # Common alias for translation function