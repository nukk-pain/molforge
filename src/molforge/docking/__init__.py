import importlib

DockingRunner = importlib.import_module("molforge.docking.module").DockingRunner

__all__ = ["DockingRunner"]
