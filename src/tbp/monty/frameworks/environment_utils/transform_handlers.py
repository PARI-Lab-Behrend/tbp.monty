import abc

from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.environment_utils.transforms import (
    AddNoiseToRawDepthImage,
    DepthTo3DLocations,
    GaussianSmoothing,
    MissingToMaxDepth,
)



class TransformHandler(metaclass=abc.ABCMeta):
    def __init__(self, next_handler=None):
        self.next_handler = next_handler

    def handle(self, obs, state):
        obs = self.call_transform(obs, state)
        if self.next_handler:
            return self.next_handler.handle(obs, state)
        return obs

    @abc.abstractmethod
    def call_transform(self, obs, state):
        pass

class MissingToMaxDepthHandler(TransformHandler):
    def __init__(self, agent_id, max_depth, threshold=0, next_handler=None):
        super().__init__(next_handler)
        self.transform = MissingToMaxDepth(agent_id, max_depth, threshold)

    def call_transform(self, obs, state=None):
        return self.transform(obs, state)

class AddNoiseHandler(TransformHandler):
    def __init__(self, agent_id, sigma, next_handler=None):
        super().__init__(next_handler)
        self.transform = AddNoiseToRawDepthImage(agent_id, sigma)
    
    def call_transform(self, obs, state=None):
        return self.transform(obs, state)

class DepthTo3DLocationsHandler(TransformHandler):
    def __init__(
        self,
        agent_id,
        sensor_ids,
        resolutions,
        zooms=1.0,
        hfov=90.0,
        clip_value=0.05,
        depth_clip_sensors=(),
        world_coord=True,
        get_all_points=False,
        use_semantic_sensor=False,
        next_handler=None,
    ):
        super().__init__(next_handler)
        self.transform = DepthTo3DLocations(
            agent_id,
            sensor_ids,
            resolutions,
            zooms,
            hfov,
            clip_value,
            depth_clip_sensors,
            world_coord,
            get_all_points,
            use_semantic_sensor,
        )
    def call_transform(self, obs, state=None):
        return self.transform(obs, state)
    

class GaussianSmoothingHandler(TransformHandler):
    def __init__(self, agent_id, sigma=2, kernel_width=3, next_handler=None):
        super().__init__(next_handler)
        self.transform = GaussianSmoothing(agent_id, sigma, kernel_width)
    def call_transform(self, obs, state=None):
        return self.transform(obs, state)
    pass


