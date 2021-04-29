# type: ignore
from . import zip_utils
from ._color_utility import (
    DEFAULT_ALL_PALETTES,
    DEFAULT_LINE_PALETTE,
    DEFAULT_PALETTE,
    ColorGradient,
    StaticColorMap,
    get_static_color_map,
    static_color_map,
)
from ._decorators import ExpectException, deprecated, enter_exit_log, suppress_error, try_catch
from .file_utility import (
    DataFrameFilenameTuple,
    create_iterator,
    default_data_folder,
    excel_to_csv,
    find_folder,
    find_parent_folder,
    find_parent_folder_with_child,
    list_filenames,
    pandas_read_csv_zip,
    pandas_to_csv_zip,
    pickle_compressed_to_file,
    pickle_to_file,
    read_excel,
    read_json,
    read_textfile,
    save_excel,
    symlink_files,
    unpickle_compressed_from_file,
    unpickle_from_file,
    write_json,
)
from .filename_fields import (
    FilenameFieldSpec,
    FilenameFieldSpecs,
    NameFieldSpecs,
    extract_filename_metadata,
    extract_filenames_metadata,
)
from .filename_utils import (
    VALID_CHARS,
    assert_that_path_exists,
    data_path_ts,
    filename_satisfied_by,
    filename_whitelist,
    filenames_satisfied_by,
    filter_names_by_pattern,
    now_timestamp,
    path_add_date,
    path_add_sequence,
    path_add_suffix,
    path_add_timestamp,
    path_of,
    replace_extension,
    replace_path,
    strip_extensions,
    strip_path_and_add_counter,
    strip_path_and_extension,
    strip_paths,
    suffix_filename,
    timestamp_filename,
    ts_data_path,
)
from .mixins import PropsMixIn
from .pandas_utils import PropertyValueMaskingOpts, create_mask, create_mask2, setup_pandas, try_split_column
from .pos_tags import (
    Known_PoS_Tag_Schemes,
    PD_PoS_tag_groups,
    PoS_Tag_Scheme,
    PoS_Tag_Schemes,
    PoS_TAGS_SCHEMES,
    get_pos_schema,
    pos_tags_to_str,
)
from .utils import (
    LOG_FORMAT,
    DummyContext,
    ListOfDicts,
    assert_is_strictly_increasing,
    better_flatten,
    chunks,
    clamp,
    clamp_values,
    complete_value_range,
    create_instance,
    dataframe_to_tuples,
    dict_of_key_values_inverted_to_dict_of_value_key,
    dict_of_lists_to_list_of_dicts,
    dict_split,
    dict_subset,
    dict_to_list_of_tuples,
    dotget,
    extend,
    extend_single,
    extract_counter_items_within_threshold,
    filter_dict,
    filter_kwargs,
    flatten,
    fn_name,
    get_func_args,
    get_logger,
    get_smiley,
    getLogger,
    ifextend,
    inspect_default_opts,
    inspect_filter_args,
    is_platform_architecture,
    is_strictly_increasing,
    isint,
    iter_windows,
    lazy_flatten,
    left_chop,
    list_of_dicts_to_dict_of_lists,
    list_to_unique_list_with_preserved_order,
    lists_of_dicts_merged_by_key,
    ls_sorted,
    multiple_replace,
    noop,
    normalize_array,
    normalize_sparse_matrix_by_vector,
    normalize_values,
    nth,
    pretty_print_matrix,
    project_series_to_range,
    project_to_range,
    project_values_to_range,
    remove_snake_case,
    right_chop,
    slim_title,
    sort_chained,
    split,
    take,
    timecall,
    timestamp,
    to_text,
    trunc_year_by,
    tuple_of_lists_to_list_of_tuples,
    uniquify,
)
from .zip_utils import compress, namelist, read, read_iterator, store
