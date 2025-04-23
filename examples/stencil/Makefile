default:
	/opt/llvm/llvm-14.x-install/bin/clang -I../../include -fopenmp -fopenmp-targets=nvptx64 -lm -lrtune -L../../src/build main.c stencil.c -o stencil


run:
	./stencil 1024


clean:
	rm stencil
