from flask import Flask
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    
    # Configuraciones (puedes definir más en config.py)
    app.config["DEBUG"] = True
    
    # Habilitar CORS si es necesario
    CORS(app)
    
    # Registrar blueprints (rutas modulares)
    from .api import api_bp
    app.register_blueprint(api_bp)
    
    return app
