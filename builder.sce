// Copyright (C) 2008 - INRIA
// Copyright (C) 2009-2011 - DIGITEO

// This file is released under the 3-clause BSD license. See COPYING-BSD.

mode(-1);
function builder_main()

  TOOLBOX_NAME  = "libsvm";
  TOOLBOX_TITLE = "libsvm Toolbox";
  toolbox_dir   = get_absolute_file_path("builder.sce");

  // Check Scilab's version
  // =============================================================================

  try
	  v = getversion("scilab");
  catch
	  error(gettext("Scilab 5.3 or more is required."));
  end

  if v(2) < 3 then
    // new API in scilab 5.3
    error(gettext('Scilab 5.3 or more is required.'));
  end
  // Check development_tools module avaibility
  // =============================================================================

if ~isdef('tbx_build_loader') then
  error(msprintf(gettext('%s module not installed."), 'modules_manager'));
end

  // Action
  // =============================================================================

  tbx_builder_macros(toolbox_dir);
  //tbx_builder_src(toolbox_dir);
  tbx_builder_gateway(toolbox_dir);
  tbx_builder_help(toolbox_dir);
  tbx_build_loader(TOOLBOX_NAME, toolbox_dir);
  tbx_build_cleaner(TOOLBOX_NAME, toolbox_dir);

endfunction 
// =============================================================================
builder_main();
clear builder_main;
// =============================================================================

