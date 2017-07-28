# The C++ program compiler.
CXX      = g++
# The pre-processor options used by the cpp (man cpp for more).
CPPFLAGS = -Wall -funroll-loops -ffast-math
# The directories in which source files reside.
SRCDIRS  = ./src
# The directories in which object files will be saved.
OBJDIRS  = ./obj
# The directories in which execute files will be saved.
BINDIRS  = ./bin
# The executable file name.
BIN      = nnlm
# The pre-processor and compiler options.
CXXFLAGS = -lm -O2

# 
SOURCES = $(foreach d,$(SRCDIRS),$(wildcard $(addprefix $(d)/*,.cpp)))
SRCS    = $(notdir $(wildcard $(SRCDIRS)/*.cpp))
OBJS    = $(addprefix $(OBJDIRS)/, $(addsuffix .o, $(basename $(SRCS))))

# create forder if not exists
define MKDIR 
if [ ! -d "$(@D)" ]; then mkdir "$(@D)" ; fi 
endef 

.PHONY: all clean

all: $(BINDIRS)/$(BIN)

# Rules for generating object files.
$(OBJDIRS)/%.o:$(SRCDIRS)/%.cpp
	@$(MKDIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

# Rules for generating the executable.
$(BINDIRS)/$(BIN):$(OBJS)
	@$(MKDIR)
	$(CXX) $(MY_CFLAGS) $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) $(OBJS) $(MY_LIBS) -o $@

clean:
	rm -f $(OBJS) $(BINDIRS)/$(BIN)