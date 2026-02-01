"""Media Engine - AI-powered video extraction API."""

try:
    from media_engine._version import __version__
except ImportError:
    # Not installed, running from source without build
    __version__ = "0.0.0.dev0"
