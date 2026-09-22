#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Recorded FM Receiver
# Description: Play a channelized 400 kSps FM baseband recording.
# GNU Radio version: 3.10.12.0

from PyQt5 import Qt
from gnuradio import qtgui
from PyQt5 import QtCore
from gnuradio import analog
from gnuradio import audio
from gnuradio import blocks
import pmt
from gnuradio import eng_notation
from gnuradio import filter
from gnuradio.filter import firdes
import sip
import threading
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation




class fm_iq(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "Recorded FM Receiver", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Recorded FM Receiver")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except BaseException as exc:
            print(f"Qt GUI: Could not set Icon: {str(exc)}", file=sys.stderr)
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("gnuradio/flowgraphs", "fm_iq")

        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)
        self.flowgraph_started = threading.Event()

        ##################################################
        # Variables
        ##################################################
        self.working_samp_rate = working_samp_rate = 400e3
        self.volume = volume = 5
        self.recorded_freq = recorded_freq = 96.3e6

        ##################################################
        # Blocks
        ##################################################

        self._volume_range = qtgui.Range(0, 11, 1, 5, 200)
        self._volume_win = qtgui.RangeWidget(self._volume_range, self.set_volume, "'volume'", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._volume_win)
        self._recorded_freq_tool_bar = Qt.QToolBar(self)
        self._recorded_freq_tool_bar.addWidget(Qt.QLabel("Recorded RF frequency" + ": "))
        self._recorded_freq_line_edit = Qt.QLineEdit(str(self.recorded_freq))
        self._recorded_freq_tool_bar.addWidget(self._recorded_freq_line_edit)
        self._recorded_freq_line_edit.editingFinished.connect(
            lambda: self.set_recorded_freq(eng_notation.str_to_num(str(self._recorded_freq_line_edit.text()))))
        self.top_layout.addWidget(self._recorded_freq_tool_bar)
        self.rational_resampler_xxx_0 = filter.rational_resampler_fff(
                interpolation=32,
                decimation=(int(working_samp_rate/1000)),
                taps=[],
                fractional_bw=0)
        self.qtgui_freq_sink_x_0 = qtgui.freq_sink_c(
            1024, #size
            window.WIN_BLACKMAN_hARRIS, #wintype
            recorded_freq, #fc
            working_samp_rate, #bw
            "", #name
            1,
            None # parent
        )
        self.qtgui_freq_sink_x_0.set_update_time(0.10)
        self.qtgui_freq_sink_x_0.set_y_axis((-140), 10)
        self.qtgui_freq_sink_x_0.set_y_label('Relative Gain', 'dB')
        self.qtgui_freq_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, 0.0, 0, "")
        self.qtgui_freq_sink_x_0.enable_autoscale(False)
        self.qtgui_freq_sink_x_0.enable_grid(False)
        self.qtgui_freq_sink_x_0.set_fft_average(1.0)
        self.qtgui_freq_sink_x_0.enable_axis_labels(True)
        self.qtgui_freq_sink_x_0.enable_control_panel(False)
        self.qtgui_freq_sink_x_0.set_fft_window_normalized(False)



        labels = ['', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "green", "black", "cyan",
            "magenta", "yellow", "dark red", "dark green", "dark blue"]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_freq_sink_x_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_freq_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_freq_sink_x_0.set_line_width(i, widths[i])
            self.qtgui_freq_sink_x_0.set_line_color(i, colors[i])
            self.qtgui_freq_sink_x_0.set_line_alpha(i, alphas[i])

        self._qtgui_freq_sink_x_0_win = sip.wrapinstance(self.qtgui_freq_sink_x_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_freq_sink_x_0_win)
        self.blocks_multiply_const_vxx_0 = blocks.multiply_const_ff((volume/10 ))
        self.blocks_file_source_0 = blocks.file_source(gr.sizeof_gr_complex*1, '/home/yshulm01/dev/yshulman/CM3070/gnuradio/fm_c96.3M_s400k/radioml_fm_30dB_s400k_7s.iq', True, 0, 0)
        self.blocks_file_source_0.set_begin_tag(pmt.PMT_NIL)
        self.audio_sink_0 = audio.sink(32000, 'pulse', True)
        self.analog_wfm_rcv_0 = analog.wfm_rcv(
        	quad_rate=working_samp_rate,
        	audio_decimation=1,
        )


        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_wfm_rcv_0, 0), (self.rational_resampler_xxx_0, 0))
        self.connect((self.blocks_file_source_0, 0), (self.analog_wfm_rcv_0, 0))
        self.connect((self.blocks_file_source_0, 0), (self.qtgui_freq_sink_x_0, 0))
        self.connect((self.blocks_multiply_const_vxx_0, 0), (self.audio_sink_0, 0))
        self.connect((self.rational_resampler_xxx_0, 0), (self.blocks_multiply_const_vxx_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("gnuradio/flowgraphs", "fm_iq")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_working_samp_rate(self):
        return self.working_samp_rate

    def set_working_samp_rate(self, working_samp_rate):
        self.working_samp_rate = working_samp_rate
        self.qtgui_freq_sink_x_0.set_frequency_range(self.recorded_freq, self.working_samp_rate)

    def get_volume(self):
        return self.volume

    def set_volume(self, volume):
        self.volume = volume
        self.blocks_multiply_const_vxx_0.set_k((self.volume/10 ))

    def get_recorded_freq(self):
        return self.recorded_freq

    def set_recorded_freq(self, recorded_freq):
        self.recorded_freq = recorded_freq
        Qt.QMetaObject.invokeMethod(self._recorded_freq_line_edit, "setText", Qt.Q_ARG("QString", eng_notation.num_to_str(self.recorded_freq)))
        self.qtgui_freq_sink_x_0.set_frequency_range(self.recorded_freq, self.working_samp_rate)




def main(top_block_cls=fm_iq, options=None):

    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()
    tb.flowgraph_started.set()

    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()

if __name__ == '__main__':
    main()
