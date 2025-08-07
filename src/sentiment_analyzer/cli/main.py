"""
Command-line interface for sentiment analysis.
"""
import sys
import json
import logging
from pathlib import Path
from typing import Optional, List

import click

from ..core.analyzer import SentimentAnalyzer
from ..core.models import ModelType, AnalysisConfig
from ..api.server import create_app

logger = logging.getLogger(__name__)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--model', '-m', type=click.Choice([m.value for m in ModelType]), 
              default=ModelType.VADER.value, help='Sentiment analysis model to use')
@click.pass_context
def cli(ctx, verbose: bool, model: str):
    """Sentiment Analyzer Pro - Advanced sentiment analysis tool."""
    # Configure logging
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create analyzer with specified model
    config = AnalysisConfig(
        model_type=ModelType(model),
        confidence_threshold=0.6,
        cache_results=True,
        enable_preprocessing=True
    )
    
    # Store in context for subcommands
    ctx.ensure_object(dict)
    ctx.obj['analyzer'] = SentimentAnalyzer(config)
    ctx.obj['verbose'] = verbose


@cli.command()
@click.argument('text')
@click.option('--output', '-o', type=click.Choice(['json', 'simple']), 
              default='simple', help='Output format')
@click.pass_context
def analyze(ctx, text: str, output: str):
    """Analyze sentiment of a single text."""
    analyzer = ctx.obj['analyzer']
    
    try:
        result = analyzer.analyze_text(text)
        
        if output == 'json':
            click.echo(json.dumps(result.to_dict(), indent=2))
        else:
            # Simple output
            emoji = {
                'positive': '😊',
                'negative': '😢', 
                'neutral': '😐'
            }.get(result.label.value, '❓')
            
            click.echo(f"{emoji} {result.label.value.upper()} (confidence: {result.confidence:.3f})")
            if ctx.obj['verbose']:
                click.echo(f"Model: {result.model_type.value}")
                click.echo(f"Processing time: {result.processing_time:.3f}s")
                click.echo(f"Scores: {result.scores}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--input', '-i', 'input_file', type=click.File('r'), 
              help='Input file with texts (one per line)')
@click.option('--output', '-o', 'output_file', type=click.File('w'), 
              help='Output file for results')
@click.option('--format', 'output_format', type=click.Choice(['json', 'csv', 'simple']), 
              default='simple', help='Output format')
@click.pass_context
def batch(ctx, input_file, output_file, output_format: str):
    """Analyze sentiment of multiple texts."""
    analyzer = ctx.obj['analyzer']
    
    # Read texts
    if input_file:
        texts = [line.strip() for line in input_file if line.strip()]
    else:
        # Read from stdin
        texts = [line.strip() for line in sys.stdin if line.strip()]
    
    if not texts:
        click.echo("No texts to analyze", err=True)
        sys.exit(1)
    
    try:
        results = analyzer.analyze_batch(texts)
        
        # Output results
        output_stream = output_file or sys.stdout
        
        if output_format == 'json':
            json.dump([result.to_dict() for result in results], output_stream, indent=2)
        elif output_format == 'csv':
            # CSV header
            print("text,label,confidence,processing_time", file=output_stream)
            for result in results:
                escaped_text = result.text.replace('"', '""')
                print(f'"{escaped_text}",{result.label.value},{result.confidence:.3f},{result.processing_time:.4f}', 
                      file=output_stream)
        else:
            # Simple format
            for i, result in enumerate(results, 1):
                emoji = {
                    'positive': '😊',
                    'negative': '😢', 
                    'neutral': '😐'
                }.get(result.label.value, '❓')
                
                print(f"{i}. {emoji} {result.label.value.upper()} ({result.confidence:.3f}) - {result.text[:60]}...", 
                      file=output_stream)
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=8000, help='Port to bind to')
@click.option('--reload', is_flag=True, help='Enable auto-reload for development')
def serve(host: str, port: int, reload: bool):
    """Start the sentiment analysis API server."""
    import uvicorn
    
    # Create app
    app = create_app()
    
    click.echo(f"Starting Sentiment Analyzer API on http://{host}:{port}")
    click.echo("API documentation available at http://{host}:{port}/docs")
    
    try:
        uvicorn.run(
            "sentiment_analyzer.api.server:create_app",
            factory=True,
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
    except KeyboardInterrupt:
        click.echo("\nShutting down server...")


@cli.command()
@click.pass_context
def models(ctx):
    """List available sentiment analysis models."""
    analyzer = ctx.obj['analyzer']
    
    model_info = analyzer.get_model_info()
    
    click.echo("Available Models:")
    click.echo("-" * 40)
    
    for model in model_info["available_models"]:
        status = "✓" if model_info["model_status"][model] else "✗"
        current = " (current)" if model == model_info["current_model"] else ""
        click.echo(f"{status} {model}{current}")


@cli.command()
@click.pass_context  
def cache(ctx):
    """Show cache statistics and management."""
    analyzer = ctx.obj['analyzer']
    
    stats = analyzer.get_cache_stats()
    
    if not stats.get("cache_enabled"):
        click.echo("Cache is disabled")
        return
    
    click.echo("Cache Statistics:")
    click.echo("-" * 30)
    click.echo(f"Size: {stats['cache_size']} items")
    click.echo(f"Hits: {stats.get('cache_hits', 0)}")
    click.echo(f"Misses: {stats.get('cache_misses', 0)}")


@cli.command()
@click.confirmation_option(prompt='Clear all cached results?')
@click.pass_context
def clear_cache(ctx):
    """Clear the analysis cache."""
    analyzer = ctx.obj['analyzer']
    analyzer.clear_cache()
    click.echo("Cache cleared successfully")


@cli.command()
@click.argument('text')
@click.pass_context
def demo(ctx, text: str):
    """Demo with all available models."""
    base_config = AnalysisConfig(
        confidence_threshold=0.6,
        cache_results=False,
        enable_preprocessing=True
    )
    
    click.echo(f"Analyzing: '{text}'")
    click.echo("=" * 50)
    
    for model_type in ModelType:
        try:
            config = AnalysisConfig(
                model_type=model_type,
                confidence_threshold=base_config.confidence_threshold,
                cache_results=base_config.cache_results,
                enable_preprocessing=base_config.enable_preprocessing
            )
            
            analyzer = SentimentAnalyzer(config)
            result = analyzer.analyze_text(text)
            
            emoji = {
                'positive': '😊',
                'negative': '😢', 
                'neutral': '😐'
            }.get(result.label.value, '❓')
            
            click.echo(f"{model_type.value:12} | {emoji} {result.label.value:8} | {result.confidence:.3f} | {result.processing_time:.3f}s")
            
        except Exception as e:
            click.echo(f"{model_type.value:12} | ❌ ERROR    | {str(e)[:50]}...")


def main():
    """Main CLI entry point."""
    cli(obj={})


if __name__ == "__main__":
    main()