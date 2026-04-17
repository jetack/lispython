import lispy  # register .lpy import hook
from lispy.lsp.server import server


def main():
    server.start_io()
