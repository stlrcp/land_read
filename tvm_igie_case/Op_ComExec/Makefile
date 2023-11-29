IGIE_ROOT=/home/zhenpeng.wang/lib/igie

PKG_CFLAGS= -std=c++14 -O3 -fPIC\
        -I${IGIE_ROOT}/include\
		-I${IGIE_ROOT}/3rdparty/dlpack/include/\
		-I${IGIE_ROOT}/3rdparty/dmlc-core/include/\
        -I/usr/include/\
        -g \
        -DDMLC_USE_LOGGING_LIBRARY=\<tvm/runtime/logging.h\>\
        # -D_GLIBCXX_USE_CXX11_ABI=0\

PKG_LDFLAGS = -L${IGIE_ROOT}/build/ -ldl -pthread


.PHONY: clean all

all: add_ComExec conv2d_ComExec cTensor_ComExec

add_ComExec: add_ComExec.cc
	@echo "IGIE_ROOT=$(IGIE_ROOT)"
	@mkdir -p $(@D)
	$(CXX) $(PKG_CFLAGS) -o $@  $^ -ltvm -lpython3.7m -I /opt/sw_home/local/cuda/include/ -L /opt/sw_home/local/cuda/lib64/ -lcuda -lcudart $(PKG_LDFLAGS)

conv2d_ComExec: conv2d_ComExec.cc
	@echo "IGIE_ROOT=$(IGIE_ROOT)"
	@mkdir -p $(@D)
	$(CXX) $(PKG_CFLAGS) -o $@  $^ -ltvm -lpython3.7m -I /opt/sw_home/local/cuda/include/ -L /opt/sw_home/local/cuda/lib64/ -lcuda -lcudart $(PKG_LDFLAGS)

cTensor_ComExec: cTensor_ComExec.cc
	@echo "IGIE_ROOT=$(IGIE_ROOT)"
	@mkdir -p $(@D)
	$(CXX) $(PKG_CFLAGS) -o $@  $^ -ltvm -lpython3.7m -I /opt/sw_home/local/cuda/include/ -L /opt/sw_home/local/cuda/lib64/ -lcuda -lcudart $(PKG_LDFLAGS)

clean:
	rm -rf add_ComExec conv2d_ComExec cTensor_ComExec
