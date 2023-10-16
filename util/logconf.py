import logging

root_logger = logging.getLogger()
root_logger.setLevel(logging.CRITICAL)  # 将根记录器的级别设置为 CRITICAL

# 移除根记录器的所有处理程序
for handler in list(root_logger.handlers):
    root_logger.removeHandler(handler)

logfmt_str = "%(asctime)s %(levelname)-8s pid:%(process)d %(name)s:%(lineno)03d:%(funcName)s %(message)s"
formatter = logging.Formatter(logfmt_str)

streamHandler = logging.StreamHandler()
streamHandler.setFormatter(formatter)
streamHandler.setLevel(logging.CRITICAL)

root_logger.addHandler(streamHandler)
