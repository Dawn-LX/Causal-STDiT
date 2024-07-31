import os

class EnvVars:
    def __init__(self) -> None:
        IS_DEBUG = os.getenv("IS_DEBUG","False").lower() in ["true","yes","1"]
        
        self.disable_all_debug = not IS_DEBUG

        self.is_turn_on = lambda env_name : os.getenv(env_name,"False").lower() in ["true","yes","1"]
    
    def __getattr__(self, name: str):
        
        if "DEBUG" in name.upper():
            if self.disable_all_debug:
                return False
            
            return os.getenv(name,"False").lower() in ["true","yes","1"]
        else:
            return os.getenv(name,"")

ENVS = EnvVars()
envs = EnvVars()

