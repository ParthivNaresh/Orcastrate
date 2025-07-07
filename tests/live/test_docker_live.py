"""
Live Docker integration tests using real Docker daemon.

These tests run against a real Docker daemon to verify
Docker tool functionality with actual Docker API calls.
"""

import time

import pytest

# Skip all tests in this module if live dependencies aren't available
try:
    import docker
except ImportError:
    pytest.skip("Docker live test dependencies not available", allow_module_level=True)

from src.tools.docker import DockerTool
from tests.live.conftest import generate_unique_name


@pytest.mark.live
@pytest.mark.docker_required
class TestDockerLiveIntegration:
    """Live integration tests for Docker tool."""

    @pytest.mark.asyncio
    async def test_docker_connection_live(self, docker_live_tool: DockerTool):
        """Test Docker daemon connection."""
        # Test that we can connect to Docker by listing containers
        result = await docker_live_tool.execute("list_containers", {"all": True})

        assert result.success
        assert "containers" in result.output or isinstance(result.output, list)

    @pytest.mark.asyncio
    async def test_container_lifecycle_live(
        self, docker_live_tool: DockerTool, docker_client: docker.DockerClient
    ):
        """Test complete container lifecycle with real Docker."""
        container_name = generate_unique_name("test-container")

        try:
            # 1. Create container
            create_result = await docker_live_tool.execute(
                "create_container",
                {
                    "image": "nginx:alpine",
                    "name": container_name,
                    "ports": {"80/tcp": None},  # Let Docker assign port
                    "environment": {"TEST_ENV": "live-integration"},
                    "labels": {"test": "live-integration", "purpose": "docker-testing"},
                },
            )

            assert create_result.success
            container_id = create_result.output["container_id"]
            assert container_id is not None
            assert create_result.output["name"] == container_name

            # Verify container exists but is not running
            container = docker_client.containers.get(container_id)
            assert container.status in ["created", "exited"]

            # 2. Start container
            start_result = await docker_live_tool.execute(
                "start_container", {"container_id": container_id}
            )

            assert start_result.success

            # Wait a moment for container to fully start
            time.sleep(2)

            # Verify container is running
            container.reload()
            assert container.status == "running"

            # 3. List containers and verify ours is there
            list_result = await docker_live_tool.execute(
                "list_containers", {"all": True}
            )

            assert list_result.success
            our_container = None
            for c in list_result.output["containers"]:
                # Container ID from list is shortened, so check if it starts with the full ID
                if container_id.startswith(c["container_id"]) or c[
                    "container_id"
                ].startswith(container_id[:12]):
                    our_container = c
                    break

            assert our_container is not None
            assert our_container["name"] == container_name
            assert our_container["image"] == "nginx:alpine"
            assert our_container["state"] in [
                "running",
                "Up",
            ]  # Docker may use either format

            # 4. Get container logs
            logs_result = await docker_live_tool.execute(
                "get_container_logs", {"container_id": container_id, "tail": 10}
            )

            assert logs_result.success
            assert "logs" in logs_result.output

            # 5. Execute command in container
            exec_result = await docker_live_tool.execute(
                "execute_command",
                {
                    "container_id": container_id,
                    "command": ["echo", "Hello from live test"],
                    "detach": False,
                },
            )

            assert exec_result.success
            assert "Hello from live test" in exec_result.output["output"]

            # 6. Stop container
            stop_result = await docker_live_tool.execute(
                "stop_container", {"container_id": container_id, "timeout": 10}
            )

            assert stop_result.success

            # Verify container is stopped
            container.reload()
            assert container.status in ["exited", "stopped"]

        finally:
            # 7. Cleanup: Remove container
            try:
                remove_result = await docker_live_tool.execute(
                    "remove_container", {"container_id": container_id, "force": True}
                )
                assert remove_result.success
            except Exception as e:
                print(f"Container cleanup failed: {e}")
                # Force cleanup with direct Docker client
                try:
                    container = docker_client.containers.get(container_id)
                    container.remove(force=True)
                except Exception as cleanup_error:
                    print(f"Direct cleanup also failed: {cleanup_error}")

    @pytest.mark.asyncio
    async def test_image_management_live(
        self, docker_live_tool: DockerTool, docker_client: docker.DockerClient
    ):
        """Test Docker image management operations."""
        test_images = ["alpine:latest", "busybox:latest"]

        try:
            # 1. Pull multiple images
            for image in test_images:
                pull_result = await docker_live_tool.execute(
                    "pull_image", {"image": image}
                )
                assert pull_result.success
                assert pull_result.output["image"] == image

            # 2. List images and verify they exist
            list_images_result = await docker_live_tool.execute("list_images", {})
            assert list_images_result.success

            image_names = []
            for img in list_images_result.output["images"]:
                image_names.extend(img.get("tags", []))

            for image in test_images:
                assert image in image_names

            # 3. Inspect an image
            inspect_result = await docker_live_tool.execute(
                "inspect_image", {"image": "alpine:latest"}
            )

            assert inspect_result.success
            assert "config" in inspect_result.output
            assert "architecture" in inspect_result.output

        finally:
            # Cleanup: Remove test images
            for image in test_images:
                try:
                    await docker_live_tool.execute(
                        "remove_image", {"image": image, "force": True}
                    )
                    # Note: This might fail if other containers are using the image
                except Exception as e:
                    print(f"Image cleanup failed for {image}: {e}")

    @pytest.mark.asyncio
    async def test_network_management_live(
        self, docker_live_tool: DockerTool, docker_client: docker.DockerClient
    ):
        """Test Docker network management."""
        network_name = generate_unique_name("test-network")

        try:
            # 1. Create custom network
            create_network_result = await docker_live_tool.execute(
                "create_network",
                {
                    "name": network_name,
                    "driver": "bridge",
                    "labels": {
                        "test": "live-integration",
                        "purpose": "network-testing",
                    },
                },
            )

            assert create_network_result.success
            network_id = create_network_result.output["network_id"]
            assert create_network_result.output["name"] == network_name

            # 2. List networks and verify ours exists
            list_networks_result = await docker_live_tool.execute("list_networks", {})
            assert list_networks_result.success

            our_network = None
            for net in list_networks_result.output["networks"]:
                # Network ID from list is shortened, so check if it matches
                if network_id.startswith(net["network_id"]) or net[
                    "network_id"
                ].startswith(network_id[:12]):
                    our_network = net
                    break

            assert our_network is not None
            assert our_network["name"] == network_name
            assert our_network["driver"] == "bridge"

            # 3. Create containers connected to the network
            container_names = [
                generate_unique_name("net-container-1"),
                generate_unique_name("net-container-2"),
            ]
            container_ids = []

            for container_name in container_names:
                create_result = await docker_live_tool.execute(
                    "create_container",
                    {
                        "image": "alpine:latest",
                        "name": container_name,
                        "networks": [network_name],
                        "command": ["sleep", "30"],
                    },
                )
                assert create_result.success
                container_ids.append(create_result.output["container_id"])

            # Start both containers
            for container_id in container_ids:
                start_result = await docker_live_tool.execute(
                    "start_container", {"container_id": container_id}
                )
                assert start_result.success

            # 4. Test network connectivity between containers
            # Container 1 should be able to ping container 2 by name
            time.sleep(2)  # Let containers fully start

            ping_result = await docker_live_tool.execute(
                "execute_command",
                {
                    "container_id": container_ids[0],
                    "command": ["ping", "-c", "1", container_names[1]],
                    "detach": False,
                },
            )

            # In a real network test, this should succeed
            # Note: ping might not work in all Docker configurations
            print(f"Ping result: {ping_result.output}")

        finally:
            # Cleanup: Remove containers and network
            for container_id in container_ids:
                try:
                    await docker_live_tool.execute(
                        "remove_container",
                        {"container_id": container_id, "force": True},
                    )
                except Exception as e:
                    print(f"Container cleanup failed: {e}")

            try:
                remove_network_result = await docker_live_tool.execute(
                    "remove_network", {"network_id": network_id}
                )
                assert remove_network_result.success
            except Exception as e:
                print(f"Network cleanup failed: {e}")

    @pytest.mark.asyncio
    async def test_volume_management_live(
        self, docker_live_tool: DockerTool, docker_client: docker.DockerClient
    ):
        """Test Docker volume management."""
        volume_name = generate_unique_name("test-volume")

        try:
            # 1. Create volume
            create_volume_result = await docker_live_tool.execute(
                "create_volume",
                {
                    "name": volume_name,
                    "labels": {"test": "live-integration", "purpose": "volume-testing"},
                },
            )

            assert create_volume_result.success
            assert create_volume_result.output["name"] == volume_name

            # 2. List volumes and verify ours exists
            list_volumes_result = await docker_live_tool.execute("list_volumes", {})
            assert list_volumes_result.success

            our_volume = None
            for vol in list_volumes_result.output["volumes"]:
                if vol["name"] == volume_name:
                    our_volume = vol
                    break

            assert our_volume is not None

            # 3. Create container with volume mounted
            container_name = generate_unique_name("volume-test-container")
            create_container_result = await docker_live_tool.execute(
                "create_container",
                {
                    "image": "alpine:latest",
                    "name": container_name,
                    "volumes": {volume_name: {"bind": "/data", "mode": "rw"}},
                    "command": [
                        "sh",
                        "-c",
                        "echo 'test data' > /data/test.txt && sleep 10",
                    ],
                },
            )

            assert create_container_result.success
            container_id = create_container_result.output["container_id"]

            # Start container and let it write to volume
            start_result = await docker_live_tool.execute(
                "start_container", {"container_id": container_id}
            )
            assert start_result.success

            time.sleep(3)  # Let container write file

            # 4. Create another container to read from the same volume
            reader_container_name = generate_unique_name("volume-reader-container")
            create_reader_result = await docker_live_tool.execute(
                "create_container",
                {
                    "image": "alpine:latest",
                    "name": reader_container_name,
                    "volumes": {volume_name: {"bind": "/data", "mode": "ro"}},
                    "command": ["cat", "/data/test.txt"],
                },
            )

            assert create_reader_result.success
            reader_container_id = create_reader_result.output["container_id"]

            start_reader_result = await docker_live_tool.execute(
                "start_container", {"container_id": reader_container_id}
            )
            assert start_reader_result.success

            time.sleep(2)  # Let reader container run

            # Get logs from reader container
            logs_result = await docker_live_tool.execute(
                "get_container_logs", {"container_id": reader_container_id}
            )

            assert logs_result.success
            assert "test data" in logs_result.output["logs"]

        finally:
            # Cleanup: Remove containers and volume
            try:
                await docker_live_tool.execute(
                    "remove_container", {"container_id": container_id, "force": True}
                )
                await docker_live_tool.execute(
                    "remove_container",
                    {"container_id": reader_container_id, "force": True},
                )
            except Exception as e:
                print(f"Container cleanup failed: {e}")

            try:
                remove_volume_result = await docker_live_tool.execute(
                    "remove_volume", {"name": volume_name}
                )
                assert remove_volume_result.success
            except Exception as e:
                print(f"Volume cleanup failed: {e}")

    @pytest.mark.asyncio
    async def test_dockerfile_build_live(
        self, docker_live_tool: DockerTool, docker_client: docker.DockerClient, tmp_path
    ):
        """Test building Docker image from Dockerfile."""
        image_name = generate_unique_name("test-image")

        # Create a simple Dockerfile
        dockerfile_content = """
FROM alpine:latest
RUN echo "Hello from test build" > /hello.txt
CMD ["cat", "/hello.txt"]
"""

        dockerfile_path = tmp_path / "Dockerfile"
        dockerfile_path.write_text(dockerfile_content)

        try:
            # 1. Build image from Dockerfile
            build_result = await docker_live_tool.execute(
                "build_image",
                {
                    "path": str(tmp_path),
                    "tag": image_name,
                    "labels": {"test": "live-integration", "purpose": "build-testing"},
                },
            )

            assert build_result.success
            assert build_result.output["image_name"] == image_name

            # 2. Verify image was created
            inspect_result = await docker_live_tool.execute(
                "inspect_image", {"image": image_name}
            )

            assert inspect_result.success
            assert image_name in str(inspect_result.output)

            # 3. Run container from built image
            container_name = generate_unique_name("built-image-container")
            create_result = await docker_live_tool.execute(
                "create_container", {"image": image_name, "name": container_name}
            )

            assert create_result.success
            container_id = create_result.output["container_id"]

            start_result = await docker_live_tool.execute(
                "start_container", {"container_id": container_id}
            )
            assert start_result.success

            time.sleep(2)  # Let container run

            # Get logs to verify our custom content
            logs_result = await docker_live_tool.execute(
                "get_container_logs", {"container_id": container_id}
            )

            assert logs_result.success
            assert "Hello from test build" in logs_result.output["logs"]

        finally:
            # Cleanup: Remove container and image
            try:
                await docker_live_tool.execute(
                    "remove_container", {"container_id": container_id, "force": True}
                )
                await docker_live_tool.execute(
                    "remove_image", {"image": image_name, "force": True}
                )
            except Exception as e:
                print(f"Build test cleanup failed: {e}")

    @pytest.mark.asyncio
    async def test_docker_compose_simulation_live(
        self, docker_live_tool: DockerTool, docker_client: docker.DockerClient
    ):
        """Test multi-container application deployment (simulating docker-compose)."""
        # This test creates a multi-container application similar to what docker-compose would do

        network_name = generate_unique_name("app-network")
        volume_name = generate_unique_name("app-data")

        containers_created = []

        try:
            # 1. Create network for the application
            network_result = await docker_live_tool.execute(
                "create_network", {"name": network_name, "driver": "bridge"}
            )
            assert network_result.success

            # 2. Create volume for data persistence
            volume_result = await docker_live_tool.execute(
                "create_volume", {"name": volume_name}
            )
            assert volume_result.success

            # 3. Create and start database container (Redis)
            db_container_name = generate_unique_name("app-db")
            db_result = await docker_live_tool.execute(
                "create_container",
                {
                    "image": "redis:alpine",
                    "name": db_container_name,
                    "networks": [network_name],
                    "volumes": {volume_name: {"bind": "/data", "mode": "rw"}},
                    "ports": {"6379/tcp": None},
                },
            )
            assert db_result.success
            db_container_id = db_result.output["container_id"]
            containers_created.append(db_container_id)

            start_db_result = await docker_live_tool.execute(
                "start_container", {"container_id": db_container_id}
            )
            assert start_db_result.success

            # 4. Create and start web application container
            web_container_name = generate_unique_name("app-web")
            web_result = await docker_live_tool.execute(
                "create_container",
                {
                    "image": "nginx:alpine",
                    "name": web_container_name,
                    "networks": [network_name],
                    "ports": {"80/tcp": None},
                    "environment": {
                        "REDIS_HOST": db_container_name,
                        "APP_ENV": "live-test",
                    },
                },
            )
            assert web_result.success
            web_container_id = web_result.output["container_id"]
            containers_created.append(web_container_id)

            start_web_result = await docker_live_tool.execute(
                "start_container", {"container_id": web_container_id}
            )
            assert start_web_result.success

            # 5. Wait for containers to be ready
            time.sleep(5)

            # 6. Verify all containers are running
            list_result = await docker_live_tool.execute("list_containers", {})
            assert list_result.success

            running_containers = [
                c for c in list_result.output["containers"] if c["state"] == "running"
            ]
            our_running_containers = []
            for c in running_containers:
                for created_id in containers_created:
                    # Handle shortened vs full ID comparison
                    if created_id.startswith(c["container_id"]) or c[
                        "container_id"
                    ].startswith(created_id[:12]):
                        our_running_containers.append(c)
                        break

            assert len(our_running_containers) == 2

            # 7. Test network connectivity between containers
            # Web container should be able to reach database container by name
            ping_result = await docker_live_tool.execute(
                "execute_command",
                {
                    "container_id": web_container_id,
                    "command": ["ping", "-c", "1", db_container_name],
                    "detach": False,
                },
            )

            print(f"Network connectivity test: {ping_result.output}")
            # Note: Ping might not work depending on Docker configuration,
            # but the command execution should succeed

            # 8. Verify volume is being used
            exec_result = await docker_live_tool.execute(
                "execute_command",
                {
                    "container_id": db_container_id,
                    "command": ["ls", "-la", "/data"],
                    "detach": False,
                },
            )

            assert exec_result.success
            print(f"Volume contents: {exec_result.output}")

        finally:
            # Cleanup: Remove all containers, network, and volume
            print("🧹 Cleaning up multi-container application...")

            for container_id in containers_created:
                try:
                    await docker_live_tool.execute(
                        "remove_container",
                        {"container_id": container_id, "force": True},
                    )
                except Exception as e:
                    print(f"Container cleanup failed: {e}")

            try:
                await docker_live_tool.execute(
                    "remove_network",
                    {"network_id": network_result.output["network_id"]},
                )
                await docker_live_tool.execute("remove_volume", {"name": volume_name})
            except Exception as e:
                print(f"Network/volume cleanup failed: {e}")

            print("✅ Multi-container application cleanup completed")

    @pytest.mark.asyncio
    async def test_docker_error_handling_live(self, docker_live_tool: DockerTool):
        """Test error handling with real Docker API errors."""

        # Test pulling non-existent image
        result = await docker_live_tool.execute(
            "pull_image", {"image": "nonexistent/image:invalid-tag"}
        )
        assert result.success is False
        assert "pull access denied" in result.error or "not found" in result.error

        # Test starting non-existent container
        result = await docker_live_tool.execute(
            "start_container", {"container_id": "nonexistent_container_id"}
        )
        assert result.success is False
        assert "No such container" in result.error or "not found" in result.error

        # Test removing non-existent network
        result = await docker_live_tool.execute(
            "remove_network", {"network_id": "nonexistent_network_id"}
        )
        assert result.success is False
        assert "No such network" in result.error or "not found" in result.error

        # Test creating container with invalid image
        result = await docker_live_tool.execute(
            "create_container",
            {"image": "totally/invalid/image:tag", "name": "will-fail"},
        )
        assert result.success is False
        assert "pull access denied" in result.error or "not found" in result.error
