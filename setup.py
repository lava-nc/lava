#!/usr/bin/env python
#   -*- coding: utf-8 -*-

from setuptools import setup
from setuptools.command.install import install as _install

class install(_install):
    def pre_install_script(self):
        pass

    def post_install_script(self):
        pass

    def run(self):
        self.pre_install_script()

        _install.run(self)

        self.post_install_script()

if __name__ == '__main__':
    setup(
        name = 'lava-nc',
        version = '0.2.0',
        description = 'A Software Framework for Neuromorphic Computing',
        long_description = '',
        long_description_content_type = None,
        classifiers = [
            'Development Status :: 3 - Alpha',
            'Programming Language :: Python'
        ],
        keywords = '',

        author = '',
        author_email = '',
        maintainer = '',
        maintainer_email = '',

        license = "['BSD-3-Clause', 'LGPL-2.1']",

        url = 'https://lava-nc.org',
        project_urls = {},

        scripts = [],
        packages = ['lava-nc'],
        namespace_packages = ['lava'],
        py_modules = [
            'lava.magma.compiler.builders.builder',
            'lava.magma.compiler.builders.interfaces',
            'lava.magma.compiler.channels.interfaces',
            'lava.magma.compiler.channels.pypychannel',
            'lava.magma.compiler.compiler',
            'lava.magma.compiler.exceptions',
            'lava.magma.compiler.exec_var',
            'lava.magma.compiler.executable',
            'lava.magma.compiler.node',
            'lava.magma.compiler.utils',
            'lava.magma.core.decorator',
            'lava.magma.core.model.c.model',
            'lava.magma.core.model.c.type',
            'lava.magma.core.model.interfaces',
            'lava.magma.core.model.model',
            'lava.magma.core.model.nc.model',
            'lava.magma.core.model.nc.type',
            'lava.magma.core.model.py.model',
            'lava.magma.core.model.py.ports',
            'lava.magma.core.model.py.type',
            'lava.magma.core.model.sub.model',
            'lava.magma.core.process.interfaces',
            'lava.magma.core.process.message_interface_enum',
            'lava.magma.core.process.ports.exceptions',
            'lava.magma.core.process.ports.ports',
            'lava.magma.core.process.ports.reduce_ops',
            'lava.magma.core.process.process',
            'lava.magma.core.process.variable',
            'lava.magma.core.resources',
            'lava.magma.core.run_conditions',
            'lava.magma.core.run_configs',
            'lava.magma.core.sync.domain',
            'lava.magma.core.sync.protocol',
            'lava.magma.core.sync.protocols.async_protocol',
            'lava.magma.core.sync.protocols.loihi_protocol',
            'lava.magma.runtime.message_infrastructure.factory',
            'lava.magma.runtime.message_infrastructure.message_infrastructure_interface',
            'lava.magma.runtime.message_infrastructure.multiprocessing',
            'lava.magma.runtime.mgmt_token_enums',
            'lava.magma.runtime.runtime',
            'lava.magma.runtime.runtime_services.enums',
            'lava.magma.runtime.runtime_services.interfaces',
            'lava.magma.runtime.runtime_services.runtime_service',
            'lava.proc.conv.models',
            'lava.proc.conv.process',
            'lava.proc.conv.utils',
            'lava.proc.dense.models',
            'lava.proc.dense.process',
            'lava.proc.io',
            'lava.proc.lif.models',
            'lava.proc.lif.process',
            'lava.proc.monitor.models',
            'lava.proc.monitor.process',
            'lava.proc.sdn.models',
            'lava.proc.sdn.process',
            'lava.utils.dataloader.mnist',
            'lava.utils.float2fixed',
            'lava.utils.profiler',
            'lava.utils.validator',
            'lava.utils.visualizer'
        ],
        entry_points = {},
        data_files = [],
        package_data = {},
        install_requires = [
            'pytest',
            'unittest2',
            'numpy',
            'matplotlib',
            'scipy'
        ],
        dependency_links = [],
        zip_safe = True,
        cmdclass = {'install': install},
        python_requires = '',
        obsoletes = [],
    )
