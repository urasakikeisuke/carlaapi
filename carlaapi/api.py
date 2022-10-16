"""api.py"""

import math
import random
from typing import Any, Dict, List, Optional, Union
from queue import Queue

import carla

from .misc import validate_type, get_item
from .constants import *
from .sensor_data import *


def check_server_health(ip_address: str, tcp_port: int, timeout: float = 5.0) -> bool:
    validate_type(ip_address, str)

    try:
        tcp_port = int(tcp_port)
        timeout = float(timeout)
    except ValueError:
        validate_type(tcp_port, int)
        validate_type(timeout, (int, float))

    client = carla.Client(host=ip_address, port=tcp_port)
    client.set_timeout(timeout)

    try:
        client.get_server_version()
    except RuntimeError:
        return False
    else:
        return True


def check_version_match(ip_address: str, tcp_port: int, timeout: float = 5.0) -> Optional[Tuple[str, str, bool]]:
    """Check and return client Python API and server verion

    Args:
        ip_address (str): Server IP Address
        tcp_port (int): Server TCP port
        timeout (float, optional): Timeout seconds. Defaults to 5.0.

    Returns:
        Optional[Tuple[bool, str, str]]: Return server version and client version as Tuple if connection established else None
    """
    validate_type(ip_address, str)

    try:
        tcp_port = int(tcp_port)
        timeout = float(timeout)
    except ValueError:
        validate_type(tcp_port, int)
        validate_type(timeout, (int, float))

    client = carla.Client(host=ip_address, port=tcp_port)
    client.set_timeout(timeout)

    try:
        client_version = client.get_client_version()
        server_version = client.get_server_version()
    except RuntimeError:
        return None
    else:
        return (server_version == client_version, server_version, client_version)


class CarlaAPI():
    def __init__(
        self,
        ip_address: str,
        tcp_port: int,
        map_name: str,
        seed: int,
        reset_settings: bool = False,
        map_layer: Optional[str] = None,
        timeout: float = 10.0,
    ) -> None:
        r"""CarlaAPI constructor

        Args:
            ip_address (str): Carla server ip address.
            tcp_port (int): Carla server TCP port.
            map_name (str): Map name of Carla world.
            seed (int): Simulation seed.
            reset_settings (bool, optional): Whether to reset world settings when the world re-loaded. Defaults to False.
            map_layer (Optional[str], optional): Map layer of Carla world. Defaults to None.
            timeout (float, optional): Connection timeout seconds. Defaults to 10.0.
        """

        validate_type(ip_address, str)
        validate_type(tcp_port, int)
        validate_type(map_name, str)
        validate_type(seed, int)
        validate_type(reset_settings, bool)
        validate_type(timeout, (int, float))

        self.__seed = seed
        random.seed(self.__seed)

        self.__client: Optional[carla.Client] = carla.Client(
            host=ip_address, port=tcp_port)
        self.__client.set_timeout(timeout)

        self.__load_world(map_name, reset_settings, map_layer)

        self.__world: carla.World = self.__get_world()
        self.__map: carla.Map = self.__get_map()
        self.__traffic_manager: carla.TrafficManager = self.__get_traffic_manager()

        self.__spawned_vehicle_actors: List[Dict[str, Any]] = []
        self.__hero_vehicle_actor_id: Optional[int] = None

        self.__pedestrian_actor_ids: List[Dict[str, int]] = []

        self.__spawned_sensor_actors: List[carla.Actor] = []

        self.__destroyed: bool = False

    def get_server_version(self) -> str:
        r"""Returns the server libcarla version

        Returns:
            str: Server version.
        """

        return self.__client.get_server_version()

    def get_client_version(self) -> str:
        r"""Returns the client libcarla version

        Returns:
            str: Client version.
        """

        return self.__client.get_client_version()

    def get_available_maps(self) -> List[str]:
        r"""Returns a list of strings containing the paths of the maps available on server

        Returns:
            List[str]: Available maps on server.
        """

        return [map_.split('/')[-1] for map_ in sorted(self.__client.get_available_maps())]

    def get_available_map_layers(self) -> List[carla.MapLayer]:
        r"""Returns a list of strings containing the paths of the map layers available on server

        Returns:
            List[carla.MapLayer]: Available map layers on server.
        """

        available_map_layers: List[carla.MapLayer] = [
            carla.MapLayer.NONE, carla.MapLayer.Buildings, carla.MapLayer.Decals,
            carla.MapLayer.Foliage, carla.MapLayer.Ground, carla.MapLayer.ParkedVehicles,
            carla.MapLayer.Particles, carla.MapLayer.Props, carla.MapLayer.StreetLights,
            carla.MapLayer.Walls, carla.MapLayer.All
        ]

        return available_map_layers

    def __load_world(self, map_name: str, reset_settings: bool, map_layer: Optional[carla.MapLayer]) -> None:
        available_maps: List[str] = self.get_available_maps()

        if map_name not in available_maps:
            raise ValueError(
                f"Specified map `{map_name}` is unavailable. Available maps are {available_maps}")  # TODO

        if map_layer is not None and "Opt" not in map_name:
            map_layer = carla.MapLayer.All

        if map_layer is None:
            map_layer = carla.MapLayer.All

        available_map_layers = self.get_available_map_layers()

        if map_layer not in available_map_layers:
            raise ValueError(
                f"Specified map layer `{map_name}` is unavailable")  # TODO

        self.__client.load_world(map_name, reset_settings, map_layer)

    def __get_world(self) -> carla.World:
        return self.__client.get_world()

    def set_world_settings(self, settings: Dict[str, Any]) -> None:
        r"""Applies settings contained in an object to the simulation running

        Args:
            settings (Dict[str, Any]): World setting dictionary
        """

        world_settings: carla.WorldSettings = self.__world.get_settings()

        world_settings.synchronous_mode = True
        world_settings.no_rendering_mode = settings[WORLD_SETTINGS_NO_RENDERING_MODE] if settings.get(
            WORLD_SETTINGS_NO_RENDERING_MODE) is not None else False
        world_settings.fixed_delta_seconds = settings[WORLD_SETTINGS_FIXED_DELTA_SECONDS] if settings.get(
            WORLD_SETTINGS_FIXED_DELTA_SECONDS) is not None else 0.1
        world_settings.substepping = settings[WORLD_SETTINGS_SUBSTEPPING] if settings.get(
            WORLD_SETTINGS_SUBSTEPPING) is not None else True
        world_settings.max_substep_delta_time = settings[WORLD_SETTINGS_MAX_SUBSTEP_DELTA_TIME] if settings.get(
            WORLD_SETTINGS_MAX_SUBSTEP_DELTA_TIME) is not None else 0.01
        world_settings.max_substeps = settings[WORLD_SETTINGS_MAX_SUBSTEPS] if settings.get(
            WORLD_SETTINGS_MAX_SUBSTEPS) is not None else 10
        world_settings.max_culling_distance = settings[WORLD_SETTINGS_MAX_CULLING_DISTANCE] if settings.get(
            WORLD_SETTINGS_MAX_CULLING_DISTANCE) is not None else 0.0
        world_settings.deterministic_ragdolls = settings[WORLD_SETTINGS_DETERMINISTIC_RAGDOLLS] if settings.get(
            WORLD_SETTINGS_DETERMINISTIC_RAGDOLLS) is not None else True
        world_settings.tile_stream_distance = settings[WORLD_SETTINGS_TILE_STREAM_DISTANCE] if settings.get(
            WORLD_SETTINGS_TILE_STREAM_DISTANCE) is not None else 80.0
        world_settings.actor_active_distance = settings[WORLD_SETTINGS_ACTOR_ACTIVE_DISTANCE] if settings.get(
            WORLD_SETTINGS_ACTOR_ACTIVE_DISTANCE) is not None else 80.0

        self.__world.apply_settings(world_settings)

    def set_weather(self, weather: carla.WeatherParameters) -> bool:
        self.__world.set_weather(weather)

        return True

    def __get_map(self) -> carla.Map:
        return self.__world.get_map()

    def __get_traffic_manager(self, tcp_port: int = 8000) -> carla.TrafficManager:
        validate_type(tcp_port, int)

        traffic_manager: carla.TrafficManager = self.__client.get_trafficmanager(
            tcp_port)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_random_device_seed(self.__seed)

        return traffic_manager

    def set_traffic_manager_settings(self, settings: Dict[str, Any]) -> None:
        r"""Applies settings contained in an object to the simulation traffic manager

        Args:
            settings (Dict[str, Any]): Traffic manager setting dictionary
        """

        self.__traffic_manager.set_global_distance_to_leading_vehicle(settings[TM_SETTINGS_GLOBAL_DISTANCE_TO_LEADING_VEHICLE] if settings.get(
            TM_SETTINGS_GLOBAL_DISTANCE_TO_LEADING_VEHICLE) is not None else 10.0)
        self.__traffic_manager.set_hybrid_physics_mode(settings[TM_SETTINGS_HYBRID_PHYSICS_MODE] if settings.get(
            TM_SETTINGS_HYBRID_PHYSICS_MODE) is not None else True)
        self.__traffic_manager.set_hybrid_physics_radius(settings[TM_SETTINGS_HYBRID_PHYSICS_RADIUS] if settings.get(
            TM_SETTINGS_HYBRID_PHYSICS_RADIUS) is not None else 80.0)
        self.__traffic_manager.set_respawn_dormant_vehicles(settings[TM_SETTINGS_RESPAWN_DORMANT_VEHICLES] if settings.get(
            TM_SETTINGS_RESPAWN_DORMANT_VEHICLES) is not None else True)
        self.__traffic_manager.global_percentage_speed_difference(settings[TM_SETTINGS_GLOBAL_PERCENTAGE_SPEED_DIFFERENCE] if settings.get(
            TM_SETTINGS_GLOBAL_PERCENTAGE_SPEED_DIFFERENCE) is not None else 10.0)

    def tick(self, tolerance: float = 10.0) -> int:
        r"""Send the tick, and give way to the server. It returns the ID of the new frame computed by the server.

        Args:
            tolerance (float, optional): Maximum time the server should wait for a tick. Defaults to 10.0.

        Raises:
            RuntimeError: Raise if spawned actors are destroyed.

        Returns:
            int: Frame id.
        """

        validate_type(tolerance, (int, float))

        if self.__destroyed:
            raise RuntimeError(
                "The actor has already been destroyed. Create a new instance to avoid unexpected problems.")  # TODO

        return self.__world.tick(seconds=tolerance)

    def __get_blueprint_library(self) -> carla.BlueprintLibrary:
        return self.__world.get_blueprint_library()

    def __get_vehicle_spawn_points(self) -> List[carla.Transform]:
        return self.__map.get_spawn_points()

    def __get_vehicle_spawn_point(self, index: Optional[int] = None) -> carla.Transform:
        spawn_points: List[carla.Transform] = self.__get_vehicle_spawn_points()

        if index is None:
            return random.choice(spawn_points)
        else:
            try:
                return spawn_points[index]
            except IndexError:
                raise IndexError(
                    f"Specified spawn point index is out of bound") from None  # TODO

    def __get_vehicle_bp(self, vehicle_name: Optional[str] = None, is_hero: bool = False, safe_spawn: bool = True) -> carla.BlueprintLibrary:
        bpl = self.__get_blueprint_library()

        if vehicle_name is None:
            filter_: str = "vehicle.*"
        else:
            filter_ = vehicle_name

        bpl = bpl.filter(filter_)

        if safe_spawn:
            bpl = [x for x in bpl if not x.id.endswith('microlino')]
            bpl = [x for x in bpl if not x.id.endswith('carlacola')]
            bpl = [x for x in bpl if not x.id.endswith('cybertruck')]
            bpl = [x for x in bpl if not x.id.endswith('t2')]
            bpl = [x for x in bpl if not x.id.endswith('sprinter')]
            bpl = [x for x in bpl if not x.id.endswith('firetruck')]
            bpl = [x for x in bpl if not x.id.endswith('ambulance')]

        if is_hero:
            bpl = [x for x in bpl if int(
                x.get_attribute('number_of_wheels')) == 4]

        bp = random.choice(bpl)

        if is_hero:
            bp.set_attribute(BP_ATTR_ROLE_NAME, ROLE_NAME_HERO)
        else:
            bp.set_attribute(BP_ATTR_ROLE_NAME, ROLE_NAME_NPC)

        if bp.has_attribute(BP_ATTR_COLOR):
            color = random.choice(bp.get_attribute(
                BP_ATTR_COLOR).recommended_values)
            bp.set_attribute(BP_ATTR_COLOR, color)

        if bp.has_attribute(BP_ATTR_DRIVER_ID):
            driver_id = random.choice(bp.get_attribute(
                BP_ATTR_DRIVER_ID).recommended_values)
            bp.set_attribute(BP_ATTR_DRIVER_ID, driver_id)

        return bp

    def __spawn_vehicle_actor(self, bp: carla.BlueprintLibrary, spawn_point: carla.Transform, auto_pilot: bool = True) -> None:
        actor: Optional[carla.Actor] = self.__world.try_spawn_actor(
            bp, spawn_point)

        if actor is not None:
            actor.set_autopilot(auto_pilot)

            actor_attrs = {
                SPAWNED_ACTORS_ACTOR_ID: actor.id,
                SPAWNED_ACTORS_BLUEPRINT: bp,
            }

            if bp.get_attribute(BP_ATTR_ROLE_NAME) == ROLE_NAME_HERO:
                actor_attrs[SPAWNED_ACTORS_ATTR] = SPAWNED_ACTORS_ATTR_HERO_VEHICLE
                self.__hero_vehicle_actor_id = actor.id

            elif bp.get_attribute(BP_ATTR_ROLE_NAME) == ROLE_NAME_NPC:
                actor_attrs[SPAWNED_ACTORS_ATTR] = SPAWNED_ACTORS_ATTR_NPC_VEHICLE

            else:
                raise ValueError()  # TODO

            self.__traffic_manager.update_vehicle_lights(actor, True)
            self.__traffic_manager.ignore_lights_percentage(actor, 1.0)
            self.__traffic_manager.ignore_signs_percentage(actor, 5.0)

            self.__spawned_vehicle_actors.append(actor_attrs)

    def spawn_hero_vehicle_actor(self, vehicle_name: Optional[str] = None, safe_spawn: bool = True) -> None:
        r"""Spawn hero vehicle actor.

        Args:
            vehicle_name (str, optional): Vehicle name. Defaults to None.
            safe_spawn (bool, optional): If set, will not spawn large body vehicle. Defaults to True.
        """

        spawn_point = self.__get_vehicle_spawn_point()
        bp = self.__get_vehicle_bp(
            vehicle_name, is_hero=True, safe_spawn=False)

        self.__spawn_vehicle_actor(bp, spawn_point, safe_spawn)

    def _get_hero_vehicle_actor_id(self) -> Optional[int]:
        return self.__hero_vehicle_actor_id

    def get_hero_vehicle_actor(self) -> Optional[carla.Actor]:
        r"""Returns a hero vehicle actor. If hero vehicle is not spawned, will return `None`

        Returns:
            Optional[carla.Actor]: Hero vehicle actor.
        """

        if self.__hero_vehicle_actor_id is None:
            return None

        return self.__world.get_actor(self._get_hero_vehicle_actor_id())

    def spawn_npc_vehicle_actors(self, n: int, vehicle_name: Optional[str] = None, safe_spawn: bool = True) -> None:
        r"""Spawn NPC vehicle actors.

        Args:
            n (int): Number of NPC vehicles.
            vehicle_name (Optional[str], optional): Vehicle name. Defaults to None.
            safe_spawn (bool, optional): If set, will not spawn large body vehicle. Defaults to True.
        """

        validate_type(n, int)

        for _ in range(n):
            spawn_point = self.__get_vehicle_spawn_point()
            bp = self.__get_vehicle_bp(vehicle_name, safe_spawn=False)

            self.__spawn_vehicle_actor(bp, spawn_point, safe_spawn)

    def get_actor_states(self, id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        r"""Returns the actor's states the client recieved during last tick.

        Args:
            id (Optional[int], optional): Actor ID. Defaults to None.

        Returns:
            Optional[Dict[str, Any]]: Actor's states
        """

        if id is None:
            return None

        actor = self.__world.get_actor(id)

        state: Dict[str, Any] = {}

        state[ACTOR_STATES_BASIC] = {
            ACTOR_STATES_BASIC_ATTRIBUTES: actor.attributes,
            ACTOR_STATES_BASIC_ID: actor.id,
            ACTOR_STATES_BASIC_IS_ALIVE: actor.is_alive,
            ACTOR_STATES_BASIC_SEMANTIC_TAGS: actor.semantic_tags,
            ACTOR_STATES_BASIC_TYPE_ID: actor.type_id,
        }

        velocity: carla.Vector3D = actor.get_velocity()
        state[ACTOR_STATES_VELOCITY] = {
            ACTOR_STATES_VELOCITY_X: velocity.x,
            ACTOR_STATES_VELOCITY_Y: velocity.y,
            ACTOR_STATES_VELOCITY_Z: velocity.z,
            ACTOR_STATES_VELOCITY_LENGTH: velocity.length(),
        }

        acceleration: carla.Vector3D = actor.get_acceleration()
        state[ACTOR_STATES_ACCELERATION] = {
            ACTOR_STATES_ACCELERATION_X: acceleration.x,
            ACTOR_STATES_ACCELERATION_Y: acceleration.y,
            ACTOR_STATES_ACCELERATION_Z: acceleration.z,
            ACTOR_STATES_ACCELERATION_LENGTH: acceleration.length(),
        }

        angular_velocity: carla.Vector3D = actor.get_acceleration()
        state[ACTOR_STATES_ANGULAR_VELOCITY] = {
            ACTOR_STATES_ANGULAR_VELOCITY_X: angular_velocity.x,
            ACTOR_STATES_ANGULAR_VELOCITY_Y: angular_velocity.y,
            ACTOR_STATES_ANGULAR_VELOCITY_Z: angular_velocity.z,
            ACTOR_STATES_ANGULAR_VELOCITY_LENGTH: angular_velocity.length(),
        }

        location: carla.Vector3D = actor.get_transform().location
        state[ACTOR_STATES_LOCATION] = {
            ACTOR_STATES_LOCATION_X: location.x,
            ACTOR_STATES_LOCATION_Y: location.y,
            ACTOR_STATES_LOCATION_Z: location.z,
        }

        rotation: carla.Vector3D = actor.get_transform().rotation
        state[ACTOR_STATES_ROTATION] = {
            ACTOR_STATES_ROTATION_ROLL: rotation.roll,
            ACTOR_STATES_ROTATION_PITCH: rotation.pitch,
            ACTOR_STATES_ROTATION_YAW: rotation.yaw,
        }

        return state

    def get_hero_vehicle_actor_states(self) -> Optional[Dict[str, Any]]:
        r"""Returns the hero's states the client recieved during last tick.

        Returns:
            Optional[Dict[str, Any]]: [description]
        """

        return self.get_actor_states(id=self._get_hero_vehicle_actor_id())

    def __get_pedestrian_bp(self, filter: str) -> carla.BlueprintLibrary:
        bpl = self.__get_blueprint_library()
        bpl = bpl.filter(filter)

        return random.choice(bpl)

    def __get_pedestrian_spawn_points(self, n: int) -> List[carla.Transform]:
        validate_type(n, int)

        spawn_points = []

        for _ in range(n):
            location = self.__world.get_random_location_from_navigation()

            spawn_point = carla.Transform()
            if location is not None:
                spawn_point.location = location
                spawn_points.append(spawn_point)

        return spawn_points

    def __spawn_pedestrian_bodys(self, spawn_points: List[carla.Transform],
                                 running_pedestrians_perc: float = 10.0) -> List[float]:
        batch = []
        _rcmd_pedestrian_speeds = []

        for spawn_point in spawn_points:
            bp = self.__get_pedestrian_bp(filter="walker.pedestrian.*")

            if bp.has_attribute(BP_ATTR_IS_INVINCIBLE):
                bp.set_attribute(BP_ATTR_IS_INVINCIBLE, IS_INVINCIBLE_FALSE)

            if bp.has_attribute(BP_ATTR_SPEED):
                if (random.random() > running_pedestrians_perc):
                    _rcmd_pedestrian_speeds.append(
                        bp.get_attribute(BP_ATTR_SPEED).recommended_values[1])
                else:
                    _rcmd_pedestrian_speeds.append(
                        bp.get_attribute(BP_ATTR_SPEED).recommended_values[2])
            else:
                _rcmd_pedestrian_speeds.append(0.0)

            batch.append(carla.command.SpawnActor(bp, spawn_point))

        responses: List[carla.command.Response] = self.__client.apply_batch_sync(
            batch, True)

        rcmd_pedestrian_speeds = []

        for i, response in enumerate(responses):
            if not response.error:
                self.__pedestrian_actor_ids.append(
                    {SPAWNED_PEDESTRIAN_BODY_ID: response.actor_id})
                rcmd_pedestrian_speeds.append(_rcmd_pedestrian_speeds[i])

        return rcmd_pedestrian_speeds

    def __spawn_pedestrian_controllers(self) -> None:
        batch = []
        bp = self.__get_pedestrian_bp(filter="controller.ai.walker")

        for pedestrian_actor_id in self.__pedestrian_actor_ids:
            batch.append(carla.command.SpawnActor(
                bp, carla.Transform(), pedestrian_actor_id[SPAWNED_PEDESTRIAN_BODY_ID]))

        responses: List[carla.command.Response] = self.__client.apply_batch_sync(
            batch, True)
        for i, response in enumerate(responses):
            if not response.error:
                self.__pedestrian_actor_ids[i][SPAWNED_PEDESTRIAN_CONTROLLER_ID] = response.actor_id

    def __spawn_pedestrians(self, pedestrian_speeds: List[float]) -> None:
        self.tick()

        for pedestrian_actor_id, pedestrian_speed in zip(self.__pedestrian_actor_ids, pedestrian_speeds):
            controller_actor: carla.Actor = self.__world.get_actor(
                pedestrian_actor_id[SPAWNED_PEDESTRIAN_CONTROLLER_ID])

            controller_actor.start()
            controller_actor.go_to_location(
                self.__world.get_random_location_from_navigation())
            controller_actor.set_max_speed(float(pedestrian_speed))

    def spawn_pedestrians(self, n: int, running_pedestrians_perc: float = 10.0) -> None:
        r"""Spawn pedestrian actors.

        Args:
            n (int): Number of pedestrians.
            running_pedestrians_perc (float): Percentages of running pedestrians to spawn. Defaults to 10.0.
        """

        self.__world.set_pedestrians_seed(self.__seed)

        spawn_points = self.__get_pedestrian_spawn_points(n)
        rcmd_pedestrian_speeds = self.__spawn_pedestrian_bodys(
            spawn_points, running_pedestrians_perc)
        self.__spawn_pedestrian_controllers()

        self.__spawn_pedestrians(rcmd_pedestrian_speeds)

    def __get_sensor_bp(self, sensor_type: str) -> Optional[List[carla.BlueprintLibrary]]:
        bpl = self.__get_blueprint_library()

        try:
            bp = bpl.find(sensor_type)
        except IndexError:
            bp = None

        return bp

    def __sensor_callback(self, data: carla.SensorData, queue: Queue, sensor_type: str) -> None:
        validate_type(queue, Queue)
        validate_type(sensor_type, str)

        processed_data: Optional[Union[numpy.ndarray, SemanticLidar_t]] = None
        if sensor_type == SENSOR_TYPE_RGB:
            processed_data = process_rgb_data(data)

        elif sensor_type == SENSOR_TYPE_DEPTH:
            processed_data = process_depth_data(data)

        elif sensor_type == SENSOR_TYPE_SEMANTIC_SEGMENTATION:
            processed_data = process_semantic_segmentation_data(data)

        elif sensor_type == SENSOR_TYPE_OPTICAL_FLOW:
            processed_data = process_optical_flow_data(data)

        elif sensor_type == SENSOR_TYPE_LIDAR_RAY_CAST:
            processed_data = process_lidar_ray_cast_data(data)

        elif sensor_type == SENSOR_TYPE_LIDAR_RAY_CAST_SEMANTIC:
            processed_data = process_lidar_ray_cast_semantic_data(data)

        elif sensor_type == SENSOR_TYPE_GNSS:
            raise NotImplementedError()  # TODO

        elif sensor_type == SENSOR_TYPE_IMU:
            raise NotImplementedError()  # TODO

        else:
            raise NotImplementedError()  # TODO

        queue.put_nowait(processed_data)

    def spawn_sensor_actor(self, parent_actor: carla.Actor, sensor_type: str,
                           sensor_definitions: Dict[str, Any]) -> Dict[str, Union[str, dict, Queue]]:
        r"""Spawn sensor actor.

        Args:
            parent_actor (carla.Actor): The parent object that the spawned actor will follow around.
            sensor_type (str): Sensor type to spawn.
            sensor_definitions (Dict[str, Any]): Definitions to spawn sensor.

        Raises:
            NameError: Raise if specified sensor type is not found.
            NameError: Raise if specified sensor type and sensor type in the definition are not match.
            IndexError: Raise if specified sensor attribute in the definition is not found.

        Returns:
            Dict[str, Union[str, dict, Queue]]: Spawned sensor data, and its contain sensor queue.
        """

        validate_type(sensor_type, str)

        bp = self.__get_sensor_bp(sensor_type)
        if bp is None:
            raise NameError(
                f"Could not find specified sensor `{sensor_type}`")  # TODO

        sensor_definitions_common = get_item(
            sensor_definitions, SENSOR_DEFINITIONS_COMMON)
        sensor_definitions_sensor_id = get_item(
            sensor_definitions_common, SENSOR_DEFINITIONS_COMMON_ID)
        sensor_definitions_sensor_type = get_item(
            sensor_definitions_common, SENSOR_DEFINITIONS_COMMON_TYPE)

        if sensor_type != sensor_definitions_sensor_type:
            raise NameError(
                "Specified sensor type and sensor type in the definition are not match")  # TODO

        sensor_definitions_transform: Dict[str, float] = get_item(
            sensor_definitions, SENSOR_DEFINITIONS_TRANSFORM)
        sensor_definitions_transform_x: float = get_item(
            sensor_definitions_transform, SENSOR_DEFINITIONS_TRANSFORM_X)
        sensor_definitions_transform_y: float = get_item(
            sensor_definitions_transform, SENSOR_DEFINITIONS_TRANSFORM_Y)
        sensor_definitions_transform_z: float = get_item(
            sensor_definitions_transform, SENSOR_DEFINITIONS_TRANSFORM_Z)
        sensor_definitions_transform_roll: float = get_item(
            sensor_definitions_transform, SENSOR_DEFINITIONS_TRANSFORM_ROLL)
        sensor_definitions_transform_pitch: float = get_item(
            sensor_definitions_transform, SENSOR_DEFINITIONS_TRANSFORM_PITCH)
        sensor_definitions_transform_yaw: float = get_item(
            sensor_definitions_transform, SENSOR_DEFINITIONS_TRANSFORM_YAW)

        carla_transform: carla.Transform = carla.Transform(
            carla.Location(x=sensor_definitions_transform_x,
                           y=sensor_definitions_transform_y, z=sensor_definitions_transform_z),
            carla.Rotation(roll=sensor_definitions_transform_roll,
                           pitch=sensor_definitions_transform_pitch, yaw=sensor_definitions_transform_yaw)
        )  # left-handed

        sensor_definitions_configs: Dict[str, Any] = get_item(
            sensor_definitions, SENSOR_DEFINITIONS_CONFIG)
        for key, value in sensor_definitions_configs.items():
            try:
                bp.set_attribute(key, str(value))
            except IndexError:
                raise IndexError(
                    f"Blueprint has not attribute `{key}`") from None  # TODO

        bp.set_attribute(BP_ATTR_SENSOR_TICK, "0.0")

        intrinsic: Optional[Dict[str, Union[int, float]]] = None
        if sensor_type in [SENSOR_TYPE_DEPTH, SENSOR_TYPE_OPTICAL_FLOW, SENSOR_TYPE_RGB]:
            width: int = bp.get_attribute(BP_ATTR_IMAGE_SIZE_X).as_int()
            height: int = bp.get_attribute(BP_ATTR_IMAGE_SIZE_Y).as_int()

            cx: float = width / 2.0
            cy: float = height / 2.0

            focal: float = width / \
                (2.0 * math.tan(bp.get_attribute(BP_ATTR_FOV).as_float() * math.pi / 360.0))

            intrinsic = {
                INTRINSIC_WIDTH: width,
                INTRINSIC_HEIGHT: height,
                INTRINSIC_CX: cx,
                INTRINSIC_CY: cy,
                INTRINSIC_FOCAL: focal,
            }

        queue: Queue = Queue(maxsize=1)

        actor: carla.Actor = self.__world.spawn_actor(
            bp, carla_transform, attach_to=parent_actor)
        actor.listen(lambda data: self.__sensor_callback(
            data, queue, sensor_type))

        self.__spawned_sensor_actors.append(actor)

        sensor_package = {
            SENSOR_PACKAGE_SENSOR_ID: sensor_definitions_sensor_id,
            SENSOR_PACKAGE_SENSOR_TYPE: sensor_definitions_sensor_type,
            SENSOR_PACKAGE_INTRINSIC: intrinsic,
            SENSOR_PACKAGE_QUEUE: queue,
        }

        return sensor_package

    def destroy_all_actors(self) -> None:
        r"""Tells the simulator to destroy all spawned actors"""

        if not self.__destroyed:
            for spawned_sensor_actor in self.__spawned_sensor_actors:
                spawned_sensor_actor.stop()
                spawned_sensor_actor.destroy()

            for spawned_vehicle_actor in self.__spawned_vehicle_actors:
                vehicle_actor = self.__world.get_actor(
                    spawned_vehicle_actor[SPAWNED_ACTORS_ACTOR_ID])
                vehicle_actor.destroy()

            for pedestrian_actor_id in self.__pedestrian_actor_ids:
                controller_actor: carla.Actor = self.__world.get_actor(
                    pedestrian_actor_id[SPAWNED_PEDESTRIAN_CONTROLLER_ID])
                controller_actor.stop()
                controller_actor.destroy()

                body_actor: carla.Actor = self.__world.get_actor(
                    pedestrian_actor_id[SPAWNED_PEDESTRIAN_BODY_ID])
                body_actor.destroy()

            self.tick()

            self.__destroyed = True
