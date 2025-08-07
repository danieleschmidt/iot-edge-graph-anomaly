"""
Multi-language translation support for sentiment analyzer.
"""
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class Language(Enum):
    """Supported languages."""
    EN = "en"      # English
    ES = "es"      # Spanish
    FR = "fr"      # French
    DE = "de"      # German
    JA = "ja"      # Japanese
    ZH = "zh"      # Chinese
    PT = "pt"      # Portuguese
    IT = "it"      # Italian
    RU = "ru"      # Russian
    AR = "ar"      # Arabic


@dataclass
class TranslationConfig:
    """Translation configuration."""
    default_language: Language = Language.EN
    fallback_language: Language = Language.EN
    auto_detect: bool = True
    case_sensitive: bool = False


class TranslationManager:
    """
    Manages translations for the sentiment analyzer.
    
    Provides multi-language support for API responses, error messages,
    and user interface elements.
    """
    
    def __init__(self, config: Optional[TranslationConfig] = None):
        self.config = config or TranslationConfig()
        self.translations: Dict[str, Dict[str, str]] = {}
        self.current_language = self.config.default_language
        
        # Load default translations
        self._load_default_translations()
        
        logger.info(f"Translation manager initialized with {len(self.translations)} languages")
    
    def _load_default_translations(self):
        """Load default built-in translations."""
        
        # English (default)
        self.translations[Language.EN.value] = {
            # API responses
            "analysis_success": "Analysis completed successfully",
            "analysis_failed": "Analysis failed",
            "invalid_input": "Invalid input provided",
            "text_required": "Text is required",
            "text_too_long": "Text exceeds maximum length",
            "model_unavailable": "Sentiment model is unavailable",
            "rate_limit_exceeded": "Rate limit exceeded",
            "server_error": "Internal server error",
            
            # Sentiment labels
            "sentiment_positive": "Positive",
            "sentiment_negative": "Negative", 
            "sentiment_neutral": "Neutral",
            
            # Security messages
            "security_blocked": "Content blocked for security reasons",
            "validation_failed": "Content validation failed",
            "suspicious_content": "Suspicious content detected",
            
            # Health and status
            "service_healthy": "Service is healthy",
            "service_unhealthy": "Service is unhealthy",
            "model_ready": "Model is ready",
            "model_loading": "Model is loading",
            
            # Cache messages
            "cache_cleared": "Cache cleared successfully",
            "cache_hit": "Result from cache",
            "cache_miss": "Result not in cache",
            
            # General
            "success": "Success",
            "error": "Error",
            "warning": "Warning",
            "info": "Information"
        }
        
        # Spanish
        self.translations[Language.ES.value] = {
            "analysis_success": "Análisis completado con éxito",
            "analysis_failed": "El análisis falló",
            "invalid_input": "Entrada inválida proporcionada",
            "text_required": "Se requiere texto",
            "text_too_long": "El texto excede la longitud máxima",
            "model_unavailable": "El modelo de sentimiento no está disponible",
            "rate_limit_exceeded": "Límite de velocidad excedido",
            "server_error": "Error interno del servidor",
            "sentiment_positive": "Positivo",
            "sentiment_negative": "Negativo",
            "sentiment_neutral": "Neutral",
            "security_blocked": "Contenido bloqueado por razones de seguridad",
            "validation_failed": "La validación del contenido falló",
            "suspicious_content": "Contenido sospechoso detectado",
            "service_healthy": "El servicio está saludable",
            "service_unhealthy": "El servicio no está saludable",
            "cache_cleared": "Caché limpiado con éxito",
            "success": "Éxito",
            "error": "Error",
            "warning": "Advertencia",
            "info": "Información"
        }
        
        # French
        self.translations[Language.FR.value] = {
            "analysis_success": "Analyse terminée avec succès",
            "analysis_failed": "L'analyse a échoué",
            "invalid_input": "Entrée invalide fournie",
            "text_required": "Le texte est requis",
            "text_too_long": "Le texte dépasse la longueur maximale",
            "model_unavailable": "Le modèle de sentiment n'est pas disponible",
            "rate_limit_exceeded": "Limite de débit dépassée",
            "server_error": "Erreur interne du serveur",
            "sentiment_positive": "Positif",
            "sentiment_negative": "Négatif",
            "sentiment_neutral": "Neutre",
            "security_blocked": "Contenu bloqué pour des raisons de sécurité",
            "validation_failed": "La validation du contenu a échoué",
            "suspicious_content": "Contenu suspect détecté",
            "service_healthy": "Le service est sain",
            "service_unhealthy": "Le service n'est pas sain",
            "cache_cleared": "Cache vidé avec succès",
            "success": "Succès",
            "error": "Erreur",
            "warning": "Avertissement",
            "info": "Information"
        }
        
        # German
        self.translations[Language.DE.value] = {
            "analysis_success": "Analyse erfolgreich abgeschlossen",
            "analysis_failed": "Analyse fehlgeschlagen",
            "invalid_input": "Ungültige Eingabe bereitgestellt",
            "text_required": "Text ist erforderlich",
            "text_too_long": "Text überschreitet maximale Länge",
            "model_unavailable": "Sentiment-Modell ist nicht verfügbar",
            "rate_limit_exceeded": "Ratenlimit überschritten",
            "server_error": "Interner Serverfehler",
            "sentiment_positive": "Positiv",
            "sentiment_negative": "Negativ",
            "sentiment_neutral": "Neutral",
            "security_blocked": "Inhalt aus Sicherheitsgründen blockiert",
            "validation_failed": "Inhaltsvalidierung fehlgeschlagen",
            "suspicious_content": "Verdächtiger Inhalt erkannt",
            "service_healthy": "Service ist gesund",
            "service_unhealthy": "Service ist ungesund",
            "cache_cleared": "Cache erfolgreich geleert",
            "success": "Erfolg",
            "error": "Fehler",
            "warning": "Warnung",
            "info": "Information"
        }
        
        # Japanese
        self.translations[Language.JA.value] = {
            "analysis_success": "分析が正常に完了しました",
            "analysis_failed": "分析に失敗しました",
            "invalid_input": "無効な入力が提供されました",
            "text_required": "テキストが必要です",
            "text_too_long": "テキストが最大長を超えています",
            "model_unavailable": "感情モデルが利用できません",
            "rate_limit_exceeded": "レート制限を超えました",
            "server_error": "内部サーバーエラー",
            "sentiment_positive": "ポジティブ",
            "sentiment_negative": "ネガティブ",
            "sentiment_neutral": "ニュートラル",
            "security_blocked": "セキュリティ上の理由でコンテンツがブロックされました",
            "validation_failed": "コンテンツの検証に失敗しました",
            "suspicious_content": "疑わしいコンテンツが検出されました",
            "service_healthy": "サービスは正常です",
            "service_unhealthy": "サービスが異常です",
            "cache_cleared": "キャッシュが正常にクリアされました",
            "success": "成功",
            "error": "エラー",
            "warning": "警告",
            "info": "情報"
        }
        
        # Chinese
        self.translations[Language.ZH.value] = {
            "analysis_success": "分析成功完成",
            "analysis_failed": "分析失败",
            "invalid_input": "提供的输入无效",
            "text_required": "需要文本",
            "text_too_long": "文本超过最大长度",
            "model_unavailable": "情感模型不可用",
            "rate_limit_exceeded": "超过速率限制",
            "server_error": "内部服务器错误",
            "sentiment_positive": "积极",
            "sentiment_negative": "消极",
            "sentiment_neutral": "中性",
            "security_blocked": "内容因安全原因被阻止",
            "validation_failed": "内容验证失败",
            "suspicious_content": "检测到可疑内容",
            "service_healthy": "服务健康",
            "service_unhealthy": "服务不健康",
            "cache_cleared": "缓存清除成功",
            "success": "成功",
            "error": "错误",
            "warning": "警告",
            "info": "信息"
        }
    
    def set_language(self, language: Language) -> None:
        """Set the current language."""
        self.current_language = language
        logger.debug(f"Language set to {language.value}")
    
    def get_language(self) -> Language:
        """Get the current language."""
        return self.current_language
    
    def translate(self, key: str, language: Optional[Language] = None) -> str:
        """
        Translate a message key to the specified or current language.
        
        Args:
            key: Translation key
            language: Target language (uses current if not specified)
            
        Returns:
            Translated message or key if not found
        """
        target_language = language or self.current_language
        
        # Try target language
        if target_language.value in self.translations:
            if key in self.translations[target_language.value]:
                return self.translations[target_language.value][key]
        
        # Fallback to default language
        if self.config.fallback_language.value in self.translations:
            if key in self.translations[self.config.fallback_language.value]:
                return self.translations[self.config.fallback_language.value][key]
        
        # Return key if no translation found
        logger.warning(f"Translation not found for key: {key}")
        return key
    
    def translate_multiple(self, keys: list, language: Optional[Language] = None) -> Dict[str, str]:
        """Translate multiple keys."""
        return {key: self.translate(key, language) for key in keys}
    
    def get_supported_languages(self) -> list:
        """Get list of supported languages."""
        return [Language(lang) for lang in self.translations.keys()]
    
    def is_language_supported(self, language: Language) -> bool:
        """Check if a language is supported."""
        return language.value in self.translations
    
    def detect_language_from_accept(self, accept_language: str) -> Language:
        """
        Detect preferred language from Accept-Language header.
        
        Args:
            accept_language: HTTP Accept-Language header value
            
        Returns:
            Detected language or default
        """
        if not accept_language:
            return self.config.default_language
        
        # Parse Accept-Language header (simplified)
        languages = []
        for lang_item in accept_language.split(','):
            if ';' in lang_item:
                lang_code = lang_item.split(';')[0].strip()
            else:
                lang_code = lang_item.strip()
            
            # Extract primary language code
            primary_code = lang_code.split('-')[0].lower()
            languages.append(primary_code)
        
        # Find first supported language
        for lang_code in languages:
            try:
                language = Language(lang_code)
                if self.is_language_supported(language):
                    return language
            except ValueError:
                continue
        
        return self.config.default_language
    
    def add_translations(self, language: Language, translations: Dict[str, str]) -> None:
        """Add or update translations for a language."""
        if language.value not in self.translations:
            self.translations[language.value] = {}
        
        self.translations[language.value].update(translations)
        logger.info(f"Added {len(translations)} translations for {language.value}")
    
    def load_translations_from_file(self, language: Language, file_path: Path) -> None:
        """Load translations from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                translations = json.load(f)
            
            self.add_translations(language, translations)
            logger.info(f"Loaded translations for {language.value} from {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to load translations from {file_path}: {e}")
            raise
    
    def export_translations(self, language: Language, file_path: Path) -> None:
        """Export translations to JSON file."""
        try:
            if language.value not in self.translations:
                raise ValueError(f"No translations found for {language.value}")
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.translations[language.value], f, 
                         ensure_ascii=False, indent=2)
            
            logger.info(f"Exported translations for {language.value} to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to export translations to {file_path}: {e}")
            raise
    
    def get_translation_stats(self) -> Dict[str, Any]:
        """Get translation statistics."""
        stats = {
            "supported_languages": len(self.translations),
            "languages": list(self.translations.keys()),
            "current_language": self.current_language.value,
            "default_language": self.config.default_language.value,
            "fallback_language": self.config.fallback_language.value
        }
        
        # Count translations per language
        for lang, trans in self.translations.items():
            stats[f"{lang}_translation_count"] = len(trans)
        
        return stats


# Global translation manager instance
translation_manager = TranslationManager()


def t(key: str, language: Optional[Language] = None) -> str:
    """Convenience function for translation."""
    return translation_manager.translate(key, language)


def set_language(language: Language) -> None:
    """Convenience function to set language."""
    translation_manager.set_language(language)