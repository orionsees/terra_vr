cmake_minimum_required(VERSION 3.8)
project(arm_vr)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
install(
  DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY urdf/
  DESTINATION share/${PROJECT_NAME}/urdf
)

install(
  DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

install(
  DIRECTORY src/
  DESTINATION share/${PROJECT_NAME}/src
  FILES_MATCHING PATTERN "*.py"
)

install(
  PROGRAMS
    src/so101_bridge.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME so101_bridge
)

install(
  PROGRAMS
    src/teleoperation.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME teleoperation
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
