"""ROS 2 integration layer for the Dev 3 costmap subsystem.

Adapters here convert ROS messages, TF and parameters into the dataclasses in
costmap_core.contracts. costmap_core itself never imports ROS.
costmap_node.py is the (skeleton) node that wires them to the core pipeline.
"""
