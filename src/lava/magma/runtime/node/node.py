# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: LGPL-2.1-or-later
# See: https://spdx.org/licenses/


# # ToDo: This needs to be re-architected. We cannot spin up processes just
# #  locally within a single Node. Processes on one node might connect to
# #  processes on another node. The entire system setup must happen inside the
# #  Runtime. The Runtime needs to allocate all nodes (right now there just
# #  happens to be one), then it creates the channels connecting processes on
# #  all nodes, then it deploys all ProcBuilders to all these nodes.
# class Node:
#     """Manages a node. It is responsible for spinning up the synchronizers (
#     one for each sync domain) and the associated process models. It also
#     creates the Mgmt channels associated with all of them so as to be able to
#     communicate the Mgmt commands"""
#     def __init__(self, actor_id: int):
#         self.id = actor_id
#         self.rton_recv_port: ty.Optional[CspRecvPort] = None
#         self.ntor_send_port: ty.Optional[CspSendPort] = None
#         self.synchronizer_processes = []
#         self.process_model_processes = []
#
#         # TODO: Abstract this out. See similar comment in runtime
#         self._smm: ty.Optional[SharedMemoryManager] = None
#
#     def start(self,
#               compiled_processes_on_node: ty.Iterable[
#               'CompiledProcessMetadata'],
#               rton_recv_port: CspRecvPort,
#               ntor_send_port: CspSendPort):
#         """Starts the Node"""
#
#         self._smm = SharedMemoryManager()
#         self._smm.start()
#
#         self.rton_recv_port, self.ntor_send_port = rton_recv_port, \
#                                                    ntor_send_port
#         self.rton_recv_port.start()
#         self.ntor_send_port.start()
#         self._spin_up_synchronizer_and_process_models(compiled_processes_on_node)
#         self._command_loop()
#
#     def _send_sync_cmd_and_get_rsp(self, cmd: MGMT_COMMAND)->ty.Iterable[
#                                                              MGMT_RESPONSE]:
#         rsps = []
#         for idx, send_port in enumerate(self.send_ports):
#             send_port.send(cmd)
#         for idx, recv_port in enumerate(self.recv_ports):
#             rsps.append(recv_port.recv())
#         return rsps
#
#     def _command_loop(self):
#         while True:
#             command = self.rton_recv_port.recv()
#             rsps = self._send_sync_cmd_and_get_rsp(command)
#             if np.array_equal(command, MGMT_COMMAND.STOP):
#                 for rsp in rsps:
#                     if not np.array_equal(rsp, MGMT_RESPONSE.TERMINATED):
#                         raise ValueError(f"Wrong Response Received : {rsp}")
#                 self.ntor_send_port.send(MGMT_RESPONSE.TERMINATED)
#                 self._join()
#                 return
#             elif np.array_equal(command, MGMT_COMMAND.PAUSE):
#                 for rsp in rsps:
#                     if not np.array_equal(rsp, MGMT_RESPONSE.PAUSED):
#                         raise ValueError(f"Wrong Response Received : {rsp}")
#                 self.ntor_send_port.send(MGMT_RESPONSE.PAUSED)
#             else:
#                 for rsp in rsps:
#                     if not np.array_equal(rsp, MGMT_RESPONSE.DONE):
#                         raise ValueError(f"Wrong Response Received : {rsp}")
#                 self.ntor_send_port.send(MGMT_RESPONSE.DONE)
#
#     # ToDo: (AW) Functions like these need better inline documentation to
#     #  explain what's going on. Alternatively, break it up into sub functions
#     #  with descriptive names. Otherwise one has to carefully read line by
#     #  line to understand how it works.
#     # ToDo: (AW) The assignment of processes to SyncDomains should have
#     #  already been figured out in the compiler. The Runtime or NodeManager
#     #  should only read the resulting NodeConfiguration and execute it.
#     # ToDo: General rule: The more global a method or attribute, the longer
#     #  the name may by. The more local the shorter its name should be. This
#     #  way you void having to create massive local variable names that cause
#     #  very long code lines.
#     def _spin_up_synchronizer_and_process_models(self,
#                                                  compiled_processes_on_node):
#         # Create map from runtime_service to processes serviced by
#         # runtime_service
#         self.send_ports = []
#         self.recv_ports = []
#         synchronizer_cls_to_compiled_process_dict = defaultdict(list)
#         for compiled_process in compiled_processes_on_node:
#             sync_domain = compiled_process.sync_domain
#             # TODO: Taking the first available runtime_service
#             synchronizer_cls = sync_domain.protocol.runtime_service[0]
#             synchronizer_cls_to_compiled_process_dict[synchronizer_cls].append(compiled_process)
#
#         # Create and connect runtime_service and processes per SyncDomain
#         for actor_id_counter, (s, p) in enumerate(
#             synchronizer_cls_to_compiled_process_dict.items()):
#             # Create system process for each Lava process and to/from sync
#             # channels
#             synchronizer_to_pm_send_ports: ty.Iterable[CspSendPort] = []
#             pm_to_synchronizer_recv_ports: ty.Iterable[CspRecvPort] = []
#             for pm_actor_id_counter, compiled_process_metadata in
#                 enumerate(p):
#                 pm_object = compiled_process_metadata.process_model()
#                 synchronizer_to_pm_sync_channel =
#                   create_pypy_mgmt_channel(smm=self._smm,
#                       name=f"stop_{pm_object.__class__.__name__}_{pm_actor_id_counter}")
#                 pm_to_synchronizer_sync_channel = create_pypy_mgmt_channel(
#                   smm=self._smm,
#                       name=f"ptos_{pm_object.__class__.__name__}_{pm_actor_id_counter}")
#
#                 synchronizer_to_pm_send_ports.append(synchronizer_to_pm_sync_channel.send_port)
#                 pm_to_synchronizer_recv_ports.append(pm_to_synchronizer_sync_channel.recv_port)
#
#                 pm = Process(target=pm_object.start,
#                              args=(synchronizer_to_pm_sync_channel.recv_port,
#                                    pm_to_synchronizer_sync_channel.send_port))
#
#                 pm.start()
#             self.process_model_processes.append(pm)
#             synchronizer_object = s(actor_id=actor_id_counter)
#
#             # Create runtime_service and channels to connect to Node
#             node_to_synchronizer_mgmt_channel = create_pypy_mgmt_channel(
#                 smm=self._smm,
#                 name=f"ntos_{synchronizer_object.__class__.__name__}"
#                      f"_{synchronizer_object.id}")
#             synchronizer_to_node_mgmt_channel = create_pypy_mgmt_channel(
#                 smm=self._smm,
#                 name=f"ston_{synchronizer_object.__class__.__name__}"
#                      f"_{synchronizer_object.id}")
#
#             sp = Process(target=synchronizer_object.start,
#                          args=(p[0].process_model.implements_protocol,
#                                node_to_synchronizer_mgmt_channel.recv_port,
#                                synchronizer_to_node_mgmt_channel.send_port,
#                                synchronizer_to_pm_send_ports,
#                                pm_to_synchronizer_recv_ports))
#             self.synchronizer_processes.append(sp)
#             node_to_synchronizer_mgmt_channel.send_port.start()
#             synchronizer_to_node_mgmt_channel.recv_port.start()
#             self.send_ports.append(node_to_synchronizer_mgmt_channel.send_port)
#             self.recv_ports.append(synchronizer_to_node_mgmt_channel.recv_port)
#             sp.start()
#             actor_id_counter += 1
#
#     def _join(self):
#         """Join all spun up ports and sub processes"""
#         for port in self.send_ports:
#             port.join()
#         for port in self.recv_ports:
#             port.join()
#         for p in self.synchronizer_processes:
#             p.join()
#         for p in self.process_model_processes:
#             p.join()
