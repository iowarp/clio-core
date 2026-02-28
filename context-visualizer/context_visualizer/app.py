"""Flask application factory for the context visualizer."""

import atexit

from flask import Flask, render_template

from . import chimaera_client


def create_app():
    app = Flask(__name__)

    # Register API blueprints
    from .api.workers import bp as workers_bp
    from .api.pools import bp as pools_bp
    from .api.config import bp as config_bp
    from .api.system import bp as system_bp
    from .api.topology import bp as topology_bp
    from .api.node import bp as node_bp

    app.register_blueprint(workers_bp, url_prefix="/api")
    app.register_blueprint(pools_bp, url_prefix="/api")
    app.register_blueprint(config_bp, url_prefix="/api")
    app.register_blueprint(system_bp, url_prefix="/api")
    app.register_blueprint(topology_bp, url_prefix="/api")
    app.register_blueprint(node_bp, url_prefix="/api")

    # Template routes
    @app.route("/")
    def topology():
        return render_template("topology.html")

    @app.route("/pools")
    def pools():
        return render_template("pools.html")

    @app.route("/config")
    def config():
        return render_template("config.html")

    @app.route("/node/<int:node_id>")
    def node(node_id):
        return render_template("node.html", node_id=node_id)

    # Clean shutdown
    atexit.register(chimaera_client.finalize)

    return app
