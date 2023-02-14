import h5py
from collections import defaultdict
from queue import Queue, Empty
from manimo.utils.parallel_processing import run_threaded_command
import tempfile
import imageio

import numpy as np

def write_dict_to_hdf5(hdf5_file, data_dict):

	for key in data_dict.keys():
		# Examine Data #
		curr_data = data_dict[key]
		if type(curr_data) == list:
			curr_data = np.array(curr_data)
		dtype = type(curr_data)

		# Unwrap If Dictionary #
		if dtype == dict:
			if key not in hdf5_file: hdf5_file.create_group(key)
			write_dict_to_hdf5(hdf5_file[key], curr_data)
			continue

		# Make Room For Data #
		if key not in hdf5_file:
			if dtype != np.ndarray: dshape = ()
			else: dtype, dshape = curr_data.dtype, curr_data.shape
			hdf5_file.create_dataset(key, (1, *dshape), maxshape=(None, *dshape), dtype=dtype)
		else:
			hdf5_file[key].resize(hdf5_file[key].shape[0] + 1, axis=0)

		# Save Data #
		hdf5_file[key][-1] = curr_data

class DataLogger:
	"""This class is used to log observations from the robot environment.
	"""
	def __init__(self, path: str, save_images: bool = True):
		self._path = path
		self._save_images = save_images
		self._hdf5_file = h5py.File(path, 'w')
		self._data_queue_dict = defaultdict(Queue)
		self._video_writers = {}
		self._video_files = {}
		self._open = True
		
		# Start HDF5 Writer Thread
		hdf5_writer = lambda data: write_dict_to_hdf5(self._hdf5_file, data)
		run_threaded_command(self._write_from_queue, args=(hdf5_writer, self._data_queue_dict['hdf5']))
	
	
	def log(self, obs: dict):
		"""Log the current observation.
		Args:
			obs (dict): The current observation.
		"""
		# obtain keys containing cam images
		cams = {key: obs[key] for key in obs.keys() if 'cam' in key}
		# save images
		if self._save_images:
			for key in cams:
				self._update_video_files(obs[key])

		# save other sensors data
		other_sensors = {key: obs[key] for key in obs.keys() if 'cam' not in key}
		self._data_queue_dict['hdf5'].put(other_sensors)

	def _write_from_queue(self, writer, queue):
		while self._open:
			try: data = queue.get(timeout=1)
			except Empty: continue
			writer(data)
			queue.task_done()

	def _update_video_files(self, cam_obs: dict):
		"""Update the video files with the current observation.
		Args:
			cam_obs (dict): The current observation.
		"""
		for video_id in cam_obs:
			# Get Frame #
			img = cam_obs[video_id]
			del cam_obs[video_id]

			# Create Writer And Buffer #
			if video_id not in self._video_writers:
				filename = self.create_video_file(video_id, '.mp4')
				self._video_writers[video_id] = imageio.get_writer(filename, macro_block_size=1)
				run_threaded_command(self._write_from_queue, args=
						(self._video_writers[video_id].append_data, self._data_queue_dict[video_id]))

			# Add Image To Queue #
			self._data_queue_dict[video_id].put(img)

	def create_video_file(self, video_id, suffix):
		temp_file = tempfile.NamedTemporaryFile(suffix=suffix)
		self._video_files[video_id] = temp_file
		return temp_file.name
	
	def finish(self):
		# Finish Remaining Jobs #
		[queue.join() for queue in self._data_queue_dict.values()]

		# Close Video Writers #
		for video_id in self._video_writers:
			self._video_writers[video_id].close()

		# Save Serialized Videos #
		for video_id in self._video_files:
			# Create Folder #
			if 'videos' not in self._hdf5_file:
				self._hdf5_file.create_group('videos')

			# Get Serialized Video #
			self._video_files[video_id].seek(0)
			serialized_video = np.asarray(self._video_files[video_id].read())

			# Save Data #
			self._hdf5_file['videos'].create_dataset(video_id, data=serialized_video)
			self._video_files[video_id].close()

		# Close File #
		self._hdf5_file.close()
		self._open = False